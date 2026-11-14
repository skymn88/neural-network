"""Compile portable C and compare ALL eight validation flights to saved metrics. TEST closed."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from actuator.generalization import generalization_data
from actuator.experiment import metrics

sys.path.insert(0, str(Path(__file__).parent / "tests"))
from test_c_export import library, c_rollout


def verify(result_folder, export_folder):
    result_folder = Path(result_folder)
    export_folder = Path(export_folder)
    r = json.loads((result_folder / "result.json").read_text())
    selected = r["selected"]
    _, groups, _, extra = generalization_data()
    val = groups["validation"] + extra
    lib = library(export_folder)
    rows = []
    for f in val:
        for component in (False, True):
            out = c_rollout(
                lib,
                f.values[:, 0],
                f.values[0, 2],
                override=f.values[:, 1] if component else None,
            )
            stages = (
                [("RAM_Component", 2, 1)]
                if component
                else [("DDV", 1, 0), ("RAM_Cascade", 2, 1), ("Angle_Cascade", 3, 2)]
            )
            for stage, yi, pi in stages:
                stat = metrics(f.values[32:, yi], out[32:, pi])
                old = next(
                    x
                    for x in r["per_flight"]
                    if x["model"] == selected
                    and x["split"] == "validation"
                    and x["flight"] == f.name
                    and x["stage"] == stage
                )
                error = abs(stat["rmse"] - old["rmse"])
                assert error < 1e-7, (f.name, stage, error)
                rows.append(
                    dict(
                        check="C portable parity",
                        model=selected,
                        flight=f.name,
                        stage=stage,
                        rmse_c=stat["rmse"],
                        rmse_python=old["rmse"],
                        rmse_difference=error,
                        fit_c=stat["fit"],
                    )
                )
        print("C PASS", f.name, flush=True)
    r.setdefault("diagnostics", []).extend(rows)
    (result_folder / "result.json").write_text(json.dumps(r, indent=2))
    with pd.ExcelWriter(
        result_folder / "comparison.xlsx",
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        pd.DataFrame(r["diagnostics"]).to_excel(
            writer, sheet_name="Diagnostics", index=False
        )
    manifest = json.loads((export_folder / "export_manifest.json").read_text())
    manifest["portable_c_validation"] = dict(
        flights=len(val),
        checks=len(rows),
        maximum_rmse_difference=max(x["rmse_difference"] for x in rows),
        compiler="GCC C99 -O2 -Wall -Wextra -Werror",
        windows_mex_tested=False,
    )
    (export_folder / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    # Shared library is Linux test output, never mislabelled as a Windows MEX.
    (export_folder / "libactuator.so").unlink()
    print(json.dumps(manifest["portable_c_validation"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("result_folder")
    p.add_argument("export_folder")
    a = p.parse_args()
    verify(a.result_folder, a.export_folder)
