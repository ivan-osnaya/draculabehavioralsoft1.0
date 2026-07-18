from pathlib import Path
import json
import pandas as pd

def save_metadata(metadata, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(output_folder / "experiment_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    pd.DataFrame([metadata]).to_csv(
        output_folder / "experiment_metadata.csv",
        index=False,
    )
