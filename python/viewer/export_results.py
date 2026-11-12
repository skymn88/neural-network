"""Export only public synthetic reference results; no training and no user data."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actuator.server import REFERENCES, plot


def export(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("index.html"), destination / "index.html")
    catalog = []
    for folder in sorted(REFERENCES.iterdir()):
        result = json.loads((folder / "result.json").read_text())
        if result["config"]["source"] not in ("bundled", "pulse"):
            raise ValueError("Public synthetic data only")
        target = destination / result["id"]
        target.mkdir(exist_ok=True)
        names = json.loads((folder / "flights.json").read_text())
        result["flights"] = names
        (target / "result.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf8"
        )
        shutil.copy2(folder / "comparison.xlsx", target / "comparison.xlsx")
        for i in range(len(names)):
            for model in result["validation_scores"]:
                points = plot(result["id"], i, model)

                # Seven significant digits are sufficient for a visual preview.
                def rounded(x):
                    if isinstance(x, float):
                        return float(f"{x:.7g}")
                    if isinstance(x, list):
                        return [rounded(v) for v in x]
                    if isinstance(x, dict):
                        return {k: rounded(v) for k, v in x.items()}
                    return x

                (target / f"flight_{i}_{model}.json").write_text(
                    json.dumps(rounded(points), separators=(",", ":")), encoding="utf8"
                )
        catalog.append(
            {
                "id": result["id"],
                "source": result["config"]["source"],
                "selected": result["selected"],
            }
        )
        print("Exported", result["config"]["source"], len(names), "flights", flush=True)
    (destination / "catalog.json").write_text(json.dumps(catalog))


if __name__ == "__main__":
    export(sys.argv[1])
