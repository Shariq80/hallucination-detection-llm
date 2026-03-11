from src.pipeline.pipeline import HallucinationPipeline

def main():
    pipeline = HallucinationPipeline("configs/default.yaml")

    claim = "The Eiffel Tower is in Paris."

    result = pipeline.verify(claim)

    print("Verification Result: ")
    print(result)

if __name__ == "__main__":
    main()