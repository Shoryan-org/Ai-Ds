import re
import json
import os
import math
from typing import List, Dict, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from context_preparation import ContextLayer


DB_PATH = "vector_db/faiss_index"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 100          # candidates pulled from each retriever
FINAL_K = 10               # results shown for the sample questions
EVAL_K = 12                # results used when computing evaluation metrics

TEST_SET_PATH = "test_set.json"


def strip_passage_prefix(text: str) -> str:
    if text.startswith("passage: "):
        text = text[len("passage: "):]
    return text


ARABIC_RANGES = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"


def tokenize(text: str) -> List[str]:
    return re.findall(rf"[a-z0-9{ARABIC_RANGES}]+", text.lower())


def contains_arabic(text: str) -> bool:
    """True if text contains any Arabic-script characters."""
    return bool(re.search(rf"[{ARABIC_RANGES}]", text))


def is_source_section(doc) -> bool:
    return doc.metadata.get("section") == "Sources"


# ---------------------------------------------------------------------------
# Arabic -> English query translation
# ---------------------------------------------------------------------------
#
# The knowledge base is English-only, so an Arabic query is at a structural
# disadvantage no matter how retrieval is tuned:
#   - BM25 corpus is built from English docs -> Arabic tokens almost never
#     match anything, so BM25 contributes ~nothing for Arabic queries.
#   - The CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) is English-only,
#     so its scores aren't meaningful for Arabic text either.
#   - Only the multilingual dense embeddings (multilingual-e5-small) are
#     actually doing useful work for Arabic today, which is why Arabic
#     metrics track noticeably below the English ones.
#
# Translating the Arabic query to English before retrieval turns it back
# into the same English-to-English search everything else was tuned for,
# so BM25 and the reranker become useful again too.
#
# The model is small enough to run on CPU and is loaded lazily (only the
# first time an Arabic query actually shows up), so English-only runs pay
# no extra startup cost.

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"

_translator = None
_translation_cache: Dict[str, str] = {}


def _get_translator():
    global _translator
    if _translator is None:
        from transformers import MarianMTModel, MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        model = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME)
        _translator = (tokenizer, model)
    return _translator


def translate_arabic_to_english(text: str) -> str:
    """Translates an Arabic query to English for retrieval purposes only
    (the original, untranslated question is still what gets shown to the
    user and sent to the LLM for the final answer — see prompt_builder.py
    Section 14 on answering in the user's language).

    Falls back to returning the original Arabic text if the translation
    model can't be loaded (missing `transformers`/`sentencepiece`, no
    network on first download, etc.) or translation fails for any other
    reason. Retrieval should degrade gracefully, not crash the request.
    """
    if not text:
        return text
    if text in _translation_cache:
        return _translation_cache[text]

    try:
        tokenizer, model = _get_translator()
        batch = tokenizer([text], return_tensors="pt",
                          padding=True, truncation=True)
        generated = model.generate(**batch, max_new_tokens=128)
        translated = tokenizer.batch_decode(
            generated, skip_special_tokens=True)[0].strip()
        if not translated:
            translated = text
    except Exception as e:
        print(
            f"WARNING: Arabic->English translation failed ({e}); "
            "falling back to the original Arabic query for retrieval."
        )
        translated = text

    _translation_cache[text] = translated
    return translated


# ---- load indices ----
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db = FAISS.load_local(DB_PATH, embeddings,
                      allow_dangerous_deserialization=True)

all_docs = list(db.docstore._dict.values())
filtered_docs = [doc for doc in all_docs if not is_source_section(doc)]
bm25_corpus = [tokenize(strip_passage_prefix(doc.page_content))
               for doc in filtered_docs]
bm25 = BM25Okapi(bm25_corpus)

reranker = CrossEncoder(RERANK_MODEL)

# Single shared instance — no need to recreate this per question.
context_preparation = ContextLayer(
    max_tokens=2000,
    max_chunks=6,
    chunk_trim_length=600,
    diversity_weight=0.3,
)


def retrieve(query: str, k: int = FINAL_K) -> List[Dict]:
    """Hybrid retrieval (dense + BM25) followed by CrossEncoder reranking.

    Arabic queries are translated to English first (see
    translate_arabic_to_english) so the same English-tuned dense+BM25+
    reranker pipeline handles every query the same way, against the same
    English-only knowledge base. Retrieved documents/citations are
    unaffected either way — only the search query itself is translated.

    Returns a list of {"doc": Document, "score": float}, best first."""
    search_query = translate_arabic_to_english(
        query) if contains_arabic(query) else query

    dense_hits = db.similarity_search_with_score(
        QUERY_PREFIX + search_query, k=CANDIDATE_K)
    dense_docs = [doc for doc, _ in dense_hits if not is_source_section(doc)]

    bm25_scores = bm25.get_scores(tokenize(search_query))
    top_idx = sorted(range(len(bm25_scores)),
                     key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATE_K]
    bm25_docs = [filtered_docs[i] for i in top_idx if bm25_scores[i] > 0]

    seen = set()
    candidates = []
    for doc in dense_docs + bm25_docs:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    if contains_arabic(search_query):
        # Translation was unavailable or failed (see
        # translate_arabic_to_english) and search_query is still Arabic.
        # RERANK_MODEL is English-only, so its scores wouldn't be
        # meaningful here — skip reranking and keep the hybrid dense+BM25
        # candidate order instead, assigning a descending synthetic score
        # so downstream code (which sorts on "score") still behaves
        # correctly. This is the same safety-net behavior as before
        # translation was added.
        n = len(candidates)
        return [
            {"doc": doc, "score": float(n - i)}
            for i, doc in enumerate(candidates[:k])
        ]

    pairs = [(search_query, strip_passage_prefix(doc.page_content))
             for doc in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, rerank_scores),
                      key=lambda x: x[1], reverse=True)

    return [{"doc": doc, "score": score} for doc, score in reranked[:k]]


# ---- evaluation metrics ----

def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        # standard log2 discount (i is 0-indexed)
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    ideal = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevance_scores, k) / idcg


def load_test_set(path: str) -> Optional[List[Dict]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(test_set: List[Dict]) -> Dict[str, float]:
    """Computes Top-1, Recall@5, Recall@10, MRR, NDCG@5, NDCG@10."""
    total = len(test_set)
    top1_acc = 0
    recall5 = 0
    recall10 = 0
    mrr_sum = 0.0
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0

    for item in test_set:
        q = item["question"]
        expected_ids = set(item["expected_chunks"])

        results = retrieve(q, k=EVAL_K)

        relevance = []
        first_rank = None
        for i, res in enumerate(results, start=1):
            # Reuses ContextLayer's chunk_id logic instead of a second
            # local copy, so retrieval evaluation and the context layer
            # always agree on what counts as "the same chunk".
            doc_id = ContextLayer.get_chunk_id(res["doc"])
            rel = 1 if doc_id in expected_ids else 0
            relevance.append(rel)
            if rel == 1 and first_rank is None:
                first_rank = i

        if results and ContextLayer.get_chunk_id(results[0]["doc"]) in expected_ids:
            top1_acc += 1

        if any(relevance[:5]):
            recall5 += 1
        if any(relevance[:10]):
            recall10 += 1

        if first_rank is not None:
            mrr_sum += 1.0 / first_rank

        ndcg5_sum += ndcg_at_k(relevance, 5)
        ndcg10_sum += ndcg_at_k(relevance, 10)

    return {
        "Top-1 Accuracy": top1_acc / total,
        "Recall@5": recall5 / total,
        "Recall@10": recall10 / total,
        "MRR": mrr_sum / total,
        "NDCG@5": ndcg5_sum / total,
        "NDCG@10": ndcg10_sum / total,
    }


def evaluate_arabic_retrieval():
    """Load an Arabic test set and print metrics."""
    arabic_path = "arabic_test_set.json"
    if not os.path.exists(arabic_path):
        print("arabic_test_set.json not found. Skipping Arabic evaluation.")
        return
    with open(arabic_path, "r", encoding="utf-8") as f:
        arabic_test_set = json.load(f)
    print("\n=== Arabic Retrieval Metrics ===")
    metrics = evaluate_retrieval(arabic_test_set)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("================================")


questions = [
    "I got a tattoo two weeks ago, can I donate blood?",
    "I have fever and cold today, should I donate?",
    "I am taking antibiotics, can I donate?",
    "My hemoglobin is low, can I donate?",
]


# ============================================================
# Only run the evaluation/demo if this script is executed directly
# ============================================================
if __name__ == "__main__":

    test_set = load_test_set(TEST_SET_PATH)
    if test_set:
        print(f"\nLoaded test set: {len(test_set)} questions.\n")
        print("=== Retrieval Metrics ===")
        metrics = evaluate_retrieval(test_set)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("=========================\n")
    else:
        print("test_set.json not found. Showing sample questions only.\n")

    # --- Arabic evaluation ---
    evaluate_arabic_retrieval()

    print("\n" + "=" * 60)
    print("Sample questions through Retrieval + Context Layer")
    print("=" * 60)

    for q in questions:
        print("\n" + "=" * 60)
        print(f"Question: {q}")

        results = retrieve(q, k=FINAL_K)

        print(f"\nRetrieved chunks ({len(results)}):")
        for i, res in enumerate(results, start=1):
            doc = res["doc"]
            source = doc.metadata.get("source_file", "unknown")
            section = doc.metadata.get("section", "None")
            print(
                f"  {i}. {source} (Section: {section}) — score={res['score']:.4f}")

        prepared = context_preparation.prepare(results)

        print(f"\nSelected chunks ({len(prepared['selected_chunks'])}):")
        for ch in prepared["selected_chunks"]:
            print(
                f"  [{ch['citation_id']}] {ch['source_file']} (Section: {ch['section']}) — score={ch['score']:.4f}")

        print("\n--- Formatted Context ---")
        print(prepared["context"])

        print("\n--- Citations ---")
        for cit in prepared["citations"]:
            print(
                f"  [{cit['citation_id']}] Source: {cit['source_file']} (Section: {cit['section']})")


#!------------------------------------------------