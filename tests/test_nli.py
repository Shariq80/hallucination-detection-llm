import src.verification.nli as nli

def run_tests():
    verifier = nli.NLIVerifier()

    print("\n-----------Entailment Tests-----------")
    entailment_examples = [
        (
            "Paris is the capital of France.",
            "Paris is located in France."
        ),
        (
            "Water freezes at 0 degrees Celsius.",
            "Water turns into ice at 0°C."
        ),
        (
            "The Earth revolves around the Sun.",
            "The Sun is orbited by the Earth."
        )
    ]

    for premise, hypythesis in entailment_examples:
        result = verifier.verify(premise, hypythesis)
        print(result)
    print("\n-----------Contradiction Tests-----------")
    contradiction_examples = [
        (
            "The Eiffel Tower is in Paris.",
            "The Eiffel Tower is in Germany."
        ),
        (
            "The sky is blue during the day.",
            "The sky is green during the day."
        ),
        (
            "Humans need oxygen to survive.",
            "Humans do not need oxygen."
        )
    ]

    for premise, hypothesis in contradiction_examples:
        result = verifier.verify(premise, hypothesis)
        print(result)
    
    print("\n-----------Neutral Tests-----------")
    neutral_examples = [
        (
            "The man is reading a book.",
            "The man enjoys mystery novels."
        ),
        (
            "Tesla produces electric vehicles.",
            "Tesla will release a flying car."
        ),
        (
            "The conference was held in New York.",
            "The speaker enjoyed the event."
        )
    ]

    for premise, hypothesis in neutral_examples:
        result = verifier.verify(premise, hypothesis)
        print(result)

if __name__ == "__main__":
    run_tests()