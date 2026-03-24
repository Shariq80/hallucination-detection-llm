from src.pipeline.pipeline import HallucinationPipeline

def main():
    pipeline = HallucinationPipeline("configs/default.yaml")

    claim = "Humans live on Mars."

    result = pipeline.verify(claim)

    print("Final Verification Result: ")
    print(result)

if __name__ == "__main__":
    main()