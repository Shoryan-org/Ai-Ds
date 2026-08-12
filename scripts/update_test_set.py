import json
import os
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "vector_db/faiss_index"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TEST_SET_PATH = "test_set.json"


def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


def build_legacy_to_new_mapping(db):
    """
    Builds a mapping from the OLD chunk_id (source-section) to the NEW chunk_id(s).
    This avoids the ambiguous split('-', 1) bug by reconstructing the exact same
    string format used in build_vector_db.py.
    """
    all_docs = list(db.docstore._dict.values())
    mapping = defaultdict(list)

    for doc in all_docs:
        source = doc.metadata.get("source_file", "unknown")
        section = doc.metadata.get("section", "None")

        # Reconstruct the EXACT legacy base ID as it was before the suffix (-2, -3)
        # This matches how build_vector_db.py sets the base_id.
        legacy_base_id = f"{source}-{section}"

        # The actual new chunk_id stored in metadata (e.g., "source-section-2")
        new_chunk_id = doc.metadata.get("chunk_id", legacy_base_id)

        mapping[legacy_base_id].append(new_chunk_id)

    return mapping


def update_test_set(test_set_path, mapping):
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    updated_count = 0
    for item in test_set:
        new_expected = []
        for expected_id in item["expected_chunks"]:
            # Direct lookup: if the expected_id matches a legacy_base_id, replace it.
            # If not (e.g., it already has a suffix), keep it as-is.
            if expected_id in mapping:
                new_expected.extend(mapping[expected_id])
            else:
                new_expected.append(expected_id)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for cid in new_expected:
            if cid not in seen:
                seen.add(cid)
                unique.append(cid)
        item["expected_chunks"] = unique
        updated_count += 1

    with open(test_set_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully updated {updated_count} items in {test_set_path}")


if __name__ == "__main__":
    if not os.path.exists(TEST_SET_PATH):
        print(f"❌ File {TEST_SET_PATH} not found.")
        exit(1)

    print("🔄 Loading FAISS index...")
    db = load_faiss_index()

    print("🔄 Building legacy-to-new chunk ID mapping...")
    mapping = build_legacy_to_new_mapping(db)

    print("🔄 Updating test_set.json...")
    update_test_set(TEST_SET_PATH, mapping)

    print("✅ Update complete. Run test_retrieval.py to verify metrics.")

