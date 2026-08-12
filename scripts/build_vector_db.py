import os
import re
import yaml
from collections import defaultdict
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import Document


KNOWLEDGE_PATH = "Knowledge"
DB_PATH = "vector_db/faiss_index"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)


def load_markdown_files() -> List[Document]:

    documents = []

    print("Loading documents...")

    for file in sorted(os.listdir(KNOWLEDGE_PATH)):

        if not file.endswith(".md"):
            continue

        path = os.path.join(KNOWLEDGE_PATH, file)

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        metadata = {}
        content = raw

        match = FRONTMATTER_RE.match(raw)

        if match:
            yaml_content = match.group(1)
            content = raw[match.end():]

            try:
                parsed = yaml.safe_load(yaml_content)
                if isinstance(parsed, dict):
                    metadata = parsed
            except yaml.YAMLError as e:
                print(f"  WARNING: could not parse frontmatter in {file}: {e}")

        if "title" in metadata:
            metadata["doc_title"] = metadata.pop("title")

        if isinstance(metadata.get("keywords"), list):
            metadata["keywords"] = ", ".join(metadata["keywords"])
        if isinstance(metadata.get("official_sources"), list):
            metadata["official_sources"] = ", ".join(
                metadata["official_sources"])

        metadata["source_file"] = file

        documents.append(
            Document(page_content=content, metadata=metadata)
        )

    print(f"Loaded {len(documents)} files")

    return documents


def split_documents(documents: List[Document]) -> List[Document]:

    print("\nSplitting documents...")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection")
        ],
        strip_headers=False,
    )

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=160,
        separators=["\n\n", "\n", ". ", " "]
    )

    final_docs: List[Document] = []

    for doc in documents:

        sections = header_splitter.split_text(doc.page_content)

        for section in sections:
            merged = dict(doc.metadata)
            merged.update(section.metadata)
            section.metadata = merged

        # Skip any section that is purely "Sources"
        sections = [s for s in sections if s.metadata.get(
            "section") != "Sources"]

        chunks = chunk_splitter.split_documents(sections)

        for chunk in chunks:
            # Inject keywords, document title, and section into the indexed
            # text to boost retrieval accuracy.
            keywords = chunk.metadata.get("keywords", "")
            doc_title = chunk.metadata.get("doc_title", "")
            section = chunk.metadata.get("section", "")
            subsection = chunk.metadata.get("subsection", "")

            prefix_parts = []
            if keywords:
                prefix_parts.append(f"Keywords: {keywords}")
            if doc_title:
                prefix_parts.append(f"Document Title: {doc_title}")
            if section:
                prefix_parts.append(f"Section: {section}")
            if subsection:
                prefix_parts.append(f"Subsection: {subsection}")

            prefix = ". ".join(prefix_parts) + ". " if prefix_parts else ""

            chunk.page_content = (
                PASSAGE_PREFIX + prefix + chunk.page_content.strip()
            )

        final_docs.extend(chunks)

    # ------------------------------------------------------------------
    # Stable chunk_id: a (source_file, section) pair is not always unique —
    # RecursiveCharacterTextSplitter can split one "##" section into several
    # physical chunks. Without a per-chunk index, anything keyed on
    # "source_file-section" (deduplication, retrieval evaluation) would
    # silently treat those distinct chunks as duplicates. The first chunk of
    # a section keeps the plain "file-section" id so existing test-set
    # references stay valid; extra pieces get a "-2", "-3", ... suffix.
    # ------------------------------------------------------------------
    counters = defaultdict(int)
    for chunk in final_docs:
        key = (chunk.metadata.get("source_file"),
               chunk.metadata.get("section"))
        counters[key] += 1
        idx = counters[key]
        base_id = f"{key[0]}-{key[1]}"
        chunk.metadata["chunk_id"] = base_id if idx == 1 else f"{base_id}-{idx}"

    print(f"Created {len(final_docs)} chunks with stable chunk_id metadata")

    return final_docs


def create_vector_db(documents: List[Document]) -> None:

    print("\nCreating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.from_documents(documents, embeddings)

    print("\nSaving FAISS index...")

    db.save_local(DB_PATH)

    print("\nDONE!")
    print("Saved:", DB_PATH)


if __name__ == "__main__":

    docs = load_markdown_files()
    chunks = split_documents(docs)
    create_vector_db(chunks)
