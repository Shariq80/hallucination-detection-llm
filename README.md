# Hallucination Detection in LLMs

System pipeline:
- Claim extraction
- Evidence retrieval
- NLI verification
- Similarity scoring
- Aggregation

## How to Run

### Automated (Recommended)
You can use the provided setup script to automatically create/activate the virtual environment, install dependencies, and run the project:

```bash
bash setup_and_run.sh
```

### Manual
1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python src/main.py
   ```
5. Setup and run
   ```bash
   ./setup_and_run.sh
   ```