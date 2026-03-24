from src.pipeline.pipeline import HallucinationPipeline
from src.evaluation.test_dataset import test_data
from pathlib import Path
import json


def save_detailed_results(config_path, all_results, metrics, results_dir="results"):

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    file_name = config_path.split("/")[-1].replace(".yaml", ".json")
    file_path = Path(results_dir) / file_name

    output = {
        "config": config_path,
        "metrics": metrics,
        "results": all_results
    }

    with open(file_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nDetailed results saved to: {file_path}")


def evaluate():

    configs = [
        "configs/exp1_baseline.yaml",
        "configs/exp2_high_recall.yaml",
        "configs/exp3_nli_focused.yaml",
        "configs/exp4_strict.yaml"
    ]

    for config_path in configs:

        print(f"\n\n========== RUNNING: {config_path} ==========")

        pipeline = HallucinationPipeline(config_path)

        TP = FP = TN = FN = 0
        all_results = []

        for item in test_data:

            result = pipeline.verify(item["claim"])

            pred = "REFUTED" if result["final_result"]["hallucinated"] else "SUPPORTED"
            true = item["label"]

            # Store FULL result
            all_results.append({
                "claim": item["claim"],
                "true_label": true,
                "predicted_label": pred,
                "details": result
            })

            # Metrics calculation
            if true == "SUPPORTED" and pred == "SUPPORTED":
                TN += 1
            elif true == "SUPPORTED" and pred == "REFUTED":
                FP += 1
            elif true == "REFUTED" and pred == "REFUTED":
                TP += 1
            elif true == "REFUTED" and pred == "SUPPORTED":
                FN += 1

        total = TP + TN + FP + FN

        accuracy = (TP + TN) / total if total else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        print("\n========= FINAL METRICS =========")
        print(f"Accuracy  : {accuracy:.3f}")
        print(f"Precision : {precision:.3f}")
        print(f"Recall    : {recall:.3f}")
        print(f"F1 Score  : {f1:.3f}")
        print("================================")

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Save EVERYTHING
        save_detailed_results(config_path, all_results, metrics)


if __name__ == "__main__":
    evaluate()