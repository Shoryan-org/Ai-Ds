## Architecture Overview

The Shoryan Blood Donation Assistant follows a modular **Retrieval-Augmented Generation (RAG)** architecture. The pipeline retrieves relevant knowledge, prepares a grounded context, generates an answer using an LLM, and verifies the final response before presenting it to the user.

```text
                              User Question
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 1. RETRIEVAL LAYER                                                   │
│    scripts/test_retrieval.py                                         │
│                                                                      │
│    • Dense Search: FAISS + multilingual-e5-small                    │
│    • Sparse Search: BM25                                             │
│    • Merge and deduplicate candidates                                │
│    • Cross-Encoder reranking: MiniLM-L-6-v2                         │
│    • Select the Top-5 most relevant chunks                           │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. CONTEXT LAYER                                                     │
│    scripts/context_preparation.py                                    │
│                                                                      │
│    • Deduplicate by chunk_id and content fingerprint                 │
│    • Sort candidates by reranking score                              │
│    • Diversify sources using MMR (diversity_weight = 0.3)            │
│    • Trim long chunks (maximum 600 characters)                       │
│    • Enforce context budget (maximum 1,500 tokens / 4 chunks)        │
│    • Build the final cleaned context                                  │
│    • Generate the citation legend                                    │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. PROMPT BUILDER                                                    │
│    generation/prompt_builder.py                                      │
│                                                                      │
│    • Inject retrieved context                                        │
│    • Inject citation legend                                          │
│    • Build system prompt                                             │
│    • Enforce scope, grounding, safety, and response style            │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. LLM PROVIDER                                                      │
│    generation/llm_providers.py                                       │
│                                                                      │
│    • Gemini 3.5 Flash                                                 │
│    • OpenRouter → Qwen3-30B-A3B                                      │
│    • OpenAI → gpt-4o-mini                                            │
│    • Ollama → qwen3:30b-a3b                                         │
│    • Automatic provider fallback                                     │
│    • Send system + user prompts                                      │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 5. ANSWER VERIFIER                                                   │
│    generation/verifier.py                                            │
│                                                                      │
│    • Validate citation markers                                       │
│    • Ensure cited sources exist in the citation legend               │
│    • Check required medical disclaimer                               │
│    • Detect invalid or unsupported citations                         │
│    • Repair or replace the response when validation fails            │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                         Final Verified Answer
                         + Citations + Disclaimer
```