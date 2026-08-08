                        User Question
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│                 RETRIEVAL LAYER                          │
│ • Dense Search (FAISS)                                   │
│ • BM25 Sparse Search                                     │
│ • Hybrid Score Fusion                                    │
│ • Cross-Encoder Reranker                                 │
│ • Retrieve Top-5 Chunks                                  │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│                  CONTEXT LAYER                           │
│ • Remove duplicate chunks                                │
│ • Source diversification                                 │
│ • Token budget management                                │
│ • Citation generation                                    │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│                 GENERATION LAYER                         │
│ • Prompt Builder                                         │
│ • Gemini 3.5 Flash                                       │
│ • Automatic fallback → Qwen3-30B-A3B                     │
│ • Answer formatting                                      │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│                 VALIDATION LAYER                         │
│ • Hallucination prevention                               │
│ • Domain restriction                                     │
│ • Safety checks                                          │
│ • Medical disclaimer                                     │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
              Professional Multilingual Response






