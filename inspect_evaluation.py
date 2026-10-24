import json
from pathlib import Path

notebook_path = Path("notebooks/07_model_evaluation.ipynb")

with notebook_path.open("r", encoding="utf-8") as file:
    notebook = json.load(file)

keywords = [
    "classification_report",
    "confusion_matrix",
    "average_precision",
    "precision_recall_curve",
    "roc_auc",
    "f1_score",
    "recall_score",
    "precision_score",
    "threshold",
    "PR-AUC",
    "AUC",
]

for index, cell in enumerate(notebook.get("cells", [])):
    source = "".join(cell.get("source", []))

    outputs = []
    for output in cell.get("outputs", []):
        if "text" in output:
            text = output["text"]
            outputs.append(
                "".join(text) if isinstance(text, list) else str(text)
            )

        data = output.get("data", {})
        if "text/plain" in data:
            text = data["text/plain"]
            outputs.append(
                "".join(text) if isinstance(text, list) else str(text)
            )

    output_text = "\n".join(outputs)
    combined = (source + "\n" + output_text).lower()

    if any(keyword.lower() in combined for keyword in keywords):
        print("\n" + "=" * 70)
        print(f"CELL {index} | {cell.get('cell_type', 'unknown')}")
        print("=" * 70)
        print("SOURCE:")
        print(source[:2500])
        print("\nOUTPUT:")
        print(output_text[:2500] or "(No saved text output)")