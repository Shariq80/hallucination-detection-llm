# src/claim_generator.py
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path

# 1. Load .env from project root
# Path(__file__).resolve().parents[1] reaches the root from src/
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# 2. Initialize the Client
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(f"GOOGLE_API_KEY not found in environment. Checked: {env_path}")

client = genai.Client(api_key=api_key)

def generate_claims(prompt: str, n_claims: int = 5):
    """
    Generate short claims for hallucination detection.
    Using gemini-3.1-flash-lite-preview for maximum speed and lowest cost during testing.
    """
    try:
        # Optimized config for testing: 
        # Lower max tokens for speed, temperature 0.8 to encourage slight hallucinations.
        config = types.GenerateContentConfig(
            max_output_tokens=250,
            temperature=0.8
        )

        # Using the specific lightweight model available to your key
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=(
                f"Task: Generate exactly {n_claims} short, one-sentence claims about '{prompt}'. "
                f"Make some claims factual and some slightly hallucinated (false). "
                f"Format: Return only the claims, one per line. No numbering, bullets, or extra text."
            ),
            config=config
        )

        # Extract text content safely
        if not response or not response.text:
            return []

        text_output = response.text
        
        # Clean up output: remove leading numbers, bullets, or dashes (e.g., "1. ", "- ", "* ")
        claims = []
        for line in text_output.split("\n"):
            clean_line = line.strip().lstrip('1234567890. *-')
            if clean_line:
                claims.append(clean_line)
        
        return claims[:n_claims]

    except Exception as e:
        if "429" in str(e):
            print("Quota limit reached. Please wait ~60 seconds before retrying.")
        elif "404" in str(e):
            print("Model ID 'gemini-3.1-flash-lite-preview' not found. Falling back to 'gemini-2.5-flash-lite'...")
            return _generate_fallback(prompt, n_claims)
        else:
            print(f"Error generating claims: {e}")
        return []

def _generate_fallback(prompt: str, n_claims: int):
    """Internal fallback to a stable lite model if the preview is unavailable."""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=f"Generate {n_claims} claims about {prompt}. One per line."
        )
        return [c.strip().lstrip('1234567890. *-') for c in response.text.split("\n") if c.strip()]
    except Exception:
        return []

# Test block
# if __name__ == "__main__":
#     test_topic = "The founding of Hyderabad"
#     print(f"Generating claims for: {test_topic}...")
#     results = generate_claims(test_topic, 3)
    
#     if results:
#         for i, claim in enumerate(results, 1):
#             print(f"{i}. {claim}")
#     else:
#         print("No claims generated. Check API Key or Internet connection.")