import re

def split_atomic_claims(claim):
    """
    Splits complex claims into smaller atomic claims based on conjunctions, commas, and semicolons.
    """
    split_regex = r',|;| and | or '
    parts = re.split(split_regex, claim, flags=re.IGNORECASE)
    atomic_claims = [p.strip() for p in parts if len(p.strip()) > 5]  # discard tiny fragments
    return atomic_claims if atomic_claims else [claim]