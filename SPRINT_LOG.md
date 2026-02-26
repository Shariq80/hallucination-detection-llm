SPRINT 01 — Foundation Setup
--------------------------------
Date: 14 Feb 2026

Completed:
- Installed required libraries
- Created GitHub repo structure
- Implemented NLI module (facebook/bart-large-mnli)
- Implemented similarity scoring module
- Implemented score aggregator
- Built pipeline orchestration
- Successfully tested end-to-end system

Issues:
- faiss-gpu installation failed on Windows
- Decided to use faiss-cpu for stability

Next Sprint:
- Integrate FEVER dataset
- Implement retrieval module
- Automate evaluation metrics

SPRINT 02 — NLI Module Implementation
--------------------------------
Date: 23 Feb 2026

Completed:
- Studied Natural Language Inference (NLI)
- Understood entailment, contradiction, neutral relationships
- Justified NLI for hallucination detection
- Implemented NLIVerifier class using facebook/bart-large-mnli
- Extracted structured probabilities
- Created clean verification API
- Implemented 9 test cases:
  - 3 Entailment
  - 3 Contradiction
  - 3 Neutral
- Prepared module for integration into:
  - Score Aggregation
  - Hallucination Decision Layer

Technical Notes:
- Model: facebook/bart-large-mnli
- Framework: PyTorch + HuggingFace Transformers
- Inference optimized with torch.no_grad()
- Device-aware (CPU/GPU compatible)
