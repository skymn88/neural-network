"""Command-line alternative to the local HTML interface."""

import argparse
from actuator.data import bundled, pulse_flights
from actuator.experiment import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", choices=["bundled", "pulse", "flightLike"], default="bundled"
    )
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--horizon", type=int, default=192)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--warm-steps", type=int, default=100)
    p.add_argument("--ddv-basis", choices=["rich", "legacy"], default="rich")
    p.add_argument("--test", action="store_true")
    p.add_argument("--methods", default="oe,neural")
    a = p.parse_args()
    flights, groups = (
        bundled()
        if a.source == "bundled"
        else pulse_flights(
            profile="flightLike" if a.source == "flightLike" else "pulse",
            seed=621 if a.source == "flightLike" else 123,
        )
    )
    r, _, _ = run(
        flights,
        groups,
        source=a.source,
        steps=a.steps,
        horizon=a.horizon,
        hidden=a.hidden,
        include_test=a.test,
        methods=a.methods.split(","),
        syntheticProfile="flightLike" if a.source == "flightLike" else "pulse",
        ddv_basis=a.ddv_basis,
        warm_steps=a.warm_steps,
    )
    print(r["selected"], r["validation_scores"], r["timings"])
