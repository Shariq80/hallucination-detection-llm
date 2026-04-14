# src/claim_generator.py
import os
import requests
from dotenv import load_dotenv
import time
import random

# Load environment variables
load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_claims(prompt: str, n_claims: int = 5):
    """
    Generate short claims using a simple, low-cost model.
    """

    max_retries = 5
    base_delay = 1

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment.")

    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "claim-generator"
            }

            data = {
                # ✅ FREE + RELIABLE MODEL
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Generate exactly {n_claims} short one-sentence claims about '{prompt}'. "
                            f"Some should be true, some slightly false. "
                            f"Return only the claims, one per line. No numbering or extra text."
                        )
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }

            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=data,
                timeout=30
            )

            if not response.ok:
                print(f"\nAttempt {attempt + 1} failed")
                print("Status Code:", response.status_code)
                print("Response:", response.text)
                response.raise_for_status()

            result = response.json()

            text_output = result["choices"][0]["message"]["content"]

            claims = [
                line.strip().lstrip('1234567890. *-')
                for line in text_output.split("\n")
                if line.strip()
            ]

            return claims[:n_claims]

        except requests.exceptions.RequestException as e:
            print(f"Error (attempt {attempt + 1}): {e}")

            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {delay:.2f} seconds...\n")
            time.sleep(delay)

            if attempt == max_retries - 1:
                print("Max retries reached.")

    return []


# Test block
# if __name__ == "__main__":
#     topic = "Happy"
#     print(f"Generating claims for: {topic}...\n")

#     results = generate_claims(topic, 3)

#     if results:
#         for i, claim in enumerate(results, 1):
#             print(f"{i}. {claim}")
#     else:
#         print("No claims generated. Check API key, model availability, or quota.")