"""
Context Layer for the Shoryan RAG chatbot.

Takes reranked retrieval candidates and turns them into a clean, LLM-ready
context block:
    1. Deduplicate (stable chunk_id + near-identical content)
    2. Sort by reranker score
    3. Diversify across source files
    4. Trim overly long chunks
    5. Fit into a token budget
    6. Format into a citeable context string

Design note: this layer intentionally has no confidence scoring, threshold,
or fallback logic. Retrieval evaluation shows the correct evidence is in the
top-5 candidates ~95% of the time (Recall@5), so the system relies on the
LLM to synthesize an answer from the best available chunks rather than
gating on a per-query confidence estimate.
"""

import re
import copy
from typing import List, Dict, Optional, Any

try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(TOKENIZER.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        # Rough fallback: ~4 characters per token.
        return len(text) // 4


class ContextLayer:
    """
    Prepares retrieved chunks for the LLM. See module docstring for the
    pipeline stages.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        max_chunks: int = 10,
        chunk_trim_length: Optional[int] = 600,
        diversity_weight: float = 0.3,  # 0 = no diversification, 1 = full diversification
    ) -> None:
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.chunk_trim_length = chunk_trim_length
        self.diversity_weight = diversity_weight

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def clean_passage_content(text: str) -> str:
        """Strip the injected retrieval prefix ('passage: ', keywords, etc.)
        so the LLM sees clean, readable text."""
        if text.startswith("passage: "):
            text = text[len("passage: "):]

        match = re.search(r'(#+ .+)', text)
        if match:
            return text[match.start():]

        prefixes_to_remove = [
            r'^Keywords: .+?\. ',
            r'^Document Title: .+?\. ',
            r'^Section: .+?\. '
        ]
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, '', text)
        return text.strip()

    @staticmethod
    def get_chunk_id(doc: Any) -> str:
        """Prefer the stable chunk_id assigned at index time (see
        build_vector_db.py). Falls back to source+section for indexes built
        before that field existed."""
        if doc.metadata.get("chunk_id"):
            return doc.metadata["chunk_id"]
        source = doc.metadata.get("source_file", "unknown")
        section = doc.metadata.get("section", "None")
        return f"{source}-{section}"

    @staticmethod
    def get_source_key(doc: Any) -> str:
        """Source file key, used for diversity."""
        return doc.metadata.get("source_file", "unknown")

    def _extract_metadata(self, doc: Any) -> Dict[str, Any]:
        return {
            "source_file": doc.metadata.get("source_file", "Unknown"),
            "section": doc.metadata.get("section", "General"),
            "doc_title": doc.metadata.get("doc_title", "Blood Donation Info"),
            "category": doc.metadata.get("category", "General"),
            "last_verified": doc.metadata.get("last_verified", "N/A"),
            "official_sources": doc.metadata.get("official_sources", ""),
        }

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop exact chunk_id repeats and near-identical content (same
        first 150 chars), keeping the highest-ranked occurrence."""
        seen_ids = set()
        seen_content = set()
        unique = []
        for ch in chunks:
            doc = ch["doc"]
            chunk_id = self.get_chunk_id(doc)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            clean_text = self.clean_passage_content(doc.page_content)
            content_key = clean_text[:150].strip().lower()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            unique.append(ch)
        return unique

    def _trim_chunk_text(self, text: str, max_length: Optional[int]) -> str:
        if max_length is None or len(text) <= max_length:
            return text
        trimmed = text[:max_length]
        last_space = trimmed.rfind(' ')
        if last_space > 0:
            trimmed = trimmed[:last_space] + " ..."
        else:
            trimmed = trimmed + " ..."
        return trimmed

    def _apply_diversity(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simplified MMR: keep the top-ranked chunk, then prefer chunks from
        sources not yet represented. `diversity_weight` controls how many
        chunks from the same source are allowed before a new source is
        required (weight=1 -> one chunk per source max; weight<=0 is a
        no-op, handled by the early return below).
        """
        if self.diversity_weight <= 0 or len(chunks) <= 1:
            return chunks

        selected = [chunks[0]]
        candidates = chunks[1:]
        used_sources = {self.get_source_key(selected[0]["doc"])}

        same_source_limit = max(
            1, round((1 - self.diversity_weight) * self.max_chunks))

        for ch in candidates:
            if len(selected) >= self.max_chunks:
                break
            source = self.get_source_key(ch["doc"])
            if source not in used_sources:
                selected.append(ch)
                used_sources.add(source)
                continue
            source_count = sum(
                1 for s in selected if self.get_source_key(s["doc"]) == source
            )
            if source_count < same_source_limit:
                selected.append(ch)

        if len(selected) < self.max_chunks:
            selected_ids = {self.get_chunk_id(s["doc"]) for s in selected}
            for ch in candidates:
                if len(selected) >= self.max_chunks:
                    break
                cid = self.get_chunk_id(ch["doc"])
                if cid not in selected_ids:
                    selected.append(ch)
                    selected_ids.add(cid)

        return selected

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def prepare(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            retrieved_chunks: list of {"doc": Document, "score": float},
                already reranked by a CrossEncoder (see retrieve() in
                test_retrieval.py).

        Returns:
            {
                "context": str,                # formatted, citeable context block
                "citations": List[Dict],        # one entry per chunk in context
                "selected_chunks": List[Dict],  # lightweight summary for logging/UI
            }
        """
        chunks = self._deduplicate(retrieved_chunks)
        chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        chunks = self._apply_diversity(chunks)

        # Trim long chunks. Never mutate ch["doc"] in place — it's the same
        # object stored in the FAISS docstore, and doing so would corrupt
        # the index's text for every future query that retrieves it.
        if self.chunk_trim_length is not None:
            for ch in chunks:
                original_text = ch["doc"].page_content
                trimmed = self._trim_chunk_text(
                    original_text, self.chunk_trim_length)
                if trimmed != original_text:
                    new_doc = copy.deepcopy(ch["doc"])
                    new_doc.page_content = trimmed
                    ch["doc"] = new_doc

        # Fit into the token budget, respecting max_chunks.
        selected: List[Dict[str, Any]] = []
        total_tokens = 0
        for ch in chunks:
            if len(selected) >= self.max_chunks:
                break
            clean_text = self.clean_passage_content(ch["doc"].page_content)
            # buffer for the citation header
            tokens = count_tokens(clean_text) + 15
            if total_tokens + tokens > self.max_tokens:
                break
            selected.append(ch)
            total_tokens += tokens

        context_parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        selected_chunks: List[Dict[str, Any]] = []

        for idx, ch in enumerate(selected, start=1):
            doc = ch["doc"]
            clean_text = self.clean_passage_content(doc.page_content)
            meta = self._extract_metadata(doc)
            meta["citation_id"] = idx
            citations.append(meta)

            section_name = meta["section"] if meta["section"] != "None" else "General"
            context_parts.append(
                f"[{idx}] Source: {meta['source_file']} (Section: {section_name})\n"
                f"    {clean_text.strip()}"
            )

            selected_chunks.append({
                "citation_id": idx,
                "chunk_id": self.get_chunk_id(doc),
                "source_file": meta["source_file"],
                "section": meta["section"],
                "score": float(ch["score"]),
                "text": clean_text.strip(),
            })

        return {
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "selected_chunks": selected_chunks,
        }
