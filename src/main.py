from pipeline import HallucinationPipeline

def main():
    pipeline = HallucinationPipeline()

    claim = "The Eiffel Tower is in Berlin."
    evidence = "The Eiffel Toer is located in Paris, France."

    result = pipeline.verify(claim, evidence)

    print("Verification Result: ")
    print(result)

if __name__ == "__main__":
    main()