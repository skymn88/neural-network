import copy
import warnings
import numpy as np
from scipy.signal import lfilter
from .data import FS
from .inference import ddv_features, predict_ddv, angle_values, ram_features, rollout

EPS = 1e-9


def fit_ddv(train, val, length=16, basis="rich", fixed_ridge=None):
    scale = max(np.std(np.concatenate([f.values[:, 0] for f in train])), EPS)
    p = 1 + (9 if basis == "rich" else 4) * (length + 1)
    G = np.zeros((p, p))
    h = np.zeros(p)
    n = 0
    for f in train:
        A = ddv_features(f.values[:, 0], scale, length, basis)[32:]
        y = f.values[32:, 1]
        G += A.T @ A
        h += A.T @ y
        n += len(y)
    rows = []
    best = None
    loss = np.inf
    R = np.eye(p)
    R[0, 0] = 0
    for ridge in [fixed_ridge] if fixed_ridge is not None else [1e-7, 1e-5, 1e-3]:
        m = {
            "scale": scale,
            "lags": length,
            "basis": basis,
            "ridge": ridge,
            "weights": np.linalg.solve(G / n + ridge * R, h / n).tolist(),
        }
        err = np.concatenate(
            [predict_ddv(m, f.values[:, 0])[32:] - f.values[32:, 1] for f in val]
        )
        score = np.sqrt(np.mean(err**2))
        rows.append(
            {
                "stage": "DDV",
                "method": "nonlinear_fir",
                "ridge": ridge,
                "validation_rmse": float(score),
            }
        )
        if score < loss:
            best = m
            loss = score
    return best, rows


def fit_angle(train, val):
    x = np.concatenate([f.values[:, 2] for f in train])
    y = np.concatenate([f.values[:, 3] for f in train])
    scale = max(np.std(x), EPS)
    best = None
    loss = np.inf
    for degree in [1, 3]:
        m = {
            "scale": scale,
            "weights": np.polynomial.polynomial.polyfit(x / scale, y, degree).tolist(),
        }
        err = np.concatenate(
            [angle_values(m, f.values[:, 2]) - f.values[:, 3] for f in val]
        )
        score = np.mean(err**2)
        if score < loss:
            best = m
            loss = score
    return best


def val_rmse(model, val):
    errors = [
        rollout(model, f.values[:, 0], f.values[0, 2])[1][32:] - f.values[32:, 2]
        for f in val
    ]
    e = np.concatenate(errors)
    return float(np.sqrt(np.mean(e**2))) if np.all(np.isfinite(e)) else float("inf")


def fit_oe(train, val, ddv, angle, check_cancel=None):
    # TRAIN DDV predictions are leave-one-flight-out when at least 3 flights exist.
    if len(train) >= 3:
        upstream = []
        for j, f in enumerate(train):
            # Fixed selected DDV ridge structure; validation never enters this fold.
            rest = [g for i, g in enumerate(train) if i != j]
            fold, _ = fit_ddv(
                rest,
                rest,
                length=ddv["lags"],
                basis=ddv.get("basis", "legacy"),
                fixed_ridge=ddv["ridge"],
            )
            upstream.append(predict_ddv(fold, f.values[:, 0]))
    else:
        warnings.warn(
            "Fewer than 3 TRAIN flights: OE uses measured DDV, not flight OOF.",
            RuntimeWarning,
        )
        upstream = [f.values[:, 1] for f in train]
    scale = max(np.std(np.concatenate(upstream)), EPS)
    length = 8
    rows = []
    best = None
    best_score = np.inf
    for beta in [0, -0.03, -0.3]:
        if check_cancel is not None:
            check_cancel()
        p = 1 + 3 * (length + 1)
        G = np.zeros((p, p))
        h = np.zeros(p)
        n = 0
        for f, u in zip(train, upstream):
            Z = ram_features(u, scale, length)
            Z[0] = 0
            a = 1 + beta / FS
            A = lfilter([1 / FS], [1, -a], Z, axis=0)
            Y = f.values[:, 2] - f.values[0, 2] * a ** np.arange(len(Z))
            A, Y = A[32:], Y[32:]
            G += A.T @ A
            h += A.T @ Y
            n += len(Y)
        s = np.sqrt(np.maximum(np.diag(G) / n, EPS))
        Q = G / np.outer(s, s) / n
        R = np.eye(p)
        R[0, 0] = 0
        for ridge in [1e-6, 1e-3, 0.1]:
            w = np.linalg.solve(Q + ridge * R, h / s / n) / s
            model = {
                "ddv": ddv,
                "angle": angle,
                "ram": {
                    "kind": "oe",
                    "beta": beta,
                    "scale": scale,
                    "lags": length,
                    "weights": w.tolist(),
                },
            }
            score = val_rmse(model, val)
            rows.append(
                {
                    "stage": "RAM",
                    "method": "OE",
                    "beta": beta,
                    "ridge": ridge,
                    "validation_rmse": score,
                }
            )
            if score < best_score:
                best_score = score
                best = model
    if best is None:
        raise ValueError("All OE candidates diverged")
    return best, rows


def fit_neural(
    train,
    val,
    ddv,
    angle,
    steps=160,
    horizon=64,
    seed=42,
    hidden=16,
    log=print,
    warm_steps=100,
    check_cancel=None,
):
    if any(len(f.time) <= horizon + 34 for f in train):
        raise ValueError(
            "Every TRAIN segment must exceed horizon + 34 samples; reduce horizon."
        )
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    full = np.concatenate([f.values for f in train])
    xmean = float(np.mean(full[:, 2]))
    xs = max(float(np.std(full[:, 2])), EPS)
    vs = max(
        float(np.std(np.concatenate([np.diff(f.values[:, 2]) * FS for f in train]))),
        EPS,
    )
    zs = max(float(np.std(full[:, 1])), EPS)
    net = torch.nn.Sequential(
        torch.nn.Linear(3, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)
    )
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    X = []
    Y = []
    arrays = []
    for f in train:
        v = np.r_[0, np.diff(f.values[:, 2]) * FS]
        inp = np.column_stack(
            [f.values[1:, 1] / zs, (f.values[:-1, 2] - xmean) / xs, v[:-1] / vs]
        )
        X.append(inp)
        Y.append((v[1:] - v[:-1]) / vs)
        arrays.append(
            (
                torch.tensor(
                    ddv_features(
                        f.values[:, 0],
                        ddv["scale"],
                        ddv["lags"],
                        ddv.get("basis", "legacy"),
                    ),
                    dtype=torch.float32,
                ),
                torch.tensor(f.values, dtype=torch.float32),
                torch.tensor(v, dtype=torch.float32),
            )
        )
    X = torch.tensor(np.concatenate(X), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(Y)[:, None], dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    for k in range(warm_steps):
        if check_cancel is not None:
            check_cancel()
        ix = torch.tensor(rng.integers(0, len(X), 2048))
        loss = torch.mean((net(X[ix]) - Y[ix]) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
        opt.step()

    def export(weights):
        return {
            "ddv": dict(ddv, weights=weights.detach().numpy().tolist()),
            "angle": angle,
            "ram": {
                "kind": "neural",
                "x_mean": xmean,
                "x_scale": xs,
                "v_scale": vs,
                "ddv_scale": zs,
                "W1": net[0].weight.detach().numpy().tolist(),
                "b1": net[0].bias.detach().numpy().tolist(),
                "W2": net[2].weight.detach().numpy().ravel().tolist(),
                "b2": float(net[2].bias.detach()[0]),
            },
        }

    dw = torch.nn.Parameter(torch.tensor(ddv["weights"], dtype=torch.float32))
    warm = export(dw)
    warm_score = val_rmse(warm, val)
    rows = [
        {
            "stage": "RAM",
            "method": "NN_teacher",
            "iteration": 0,
            "validation_rmse": warm_score,
        }
    ]
    candidates = {"NN_teacher": warm}
    best = copy.deepcopy(warm)
    best_score = warm_score
    best_iteration = 0
    opt = torch.optim.Adam(
        [{"params": net.parameters(), "lr": 0.001}, {"params": [dw], "lr": 0.00003}]
    )
    for step in range(steps):
        if check_cancel is not None:
            check_cancel()
        # Windows only from TRAIN. Real RAM/velocity initialize TRAIN windows only.
        us = []
        ys = []
        v0 = []
        for b in range(24):
            fi = int(rng.integers(len(arrays)))
            features, truth, velocity = arrays[fi]
            start = int(rng.integers(33, len(truth) - horizon))
            us.append(features[start : start + horizon])
            ys.append(truth[start - 1 : start + horizon])
            v0.append(velocity[start - 1])
        U = torch.stack(us)
        T = torch.stack(ys)
        x = T[:, 0, 2]
        v = torch.stack(v0)
        z = U @ dw
        position = 0.0
        speed = 0.0
        hold = 0.0
        hold_n = 0.0
        for k in range(horizon):
            inp = torch.stack([z[:, k] / zs, (x - xmean) / xs, v / vs], dim=1)
            v = (v / vs + net(inp).squeeze(1)) * vs
            x = x + v / FS
            target_v = (T[:, k + 1, 2] - T[:, k, 2]) * FS
            position = position + torch.mean(((x - T[:, k + 1, 2]) / xs) ** 2)
            speed = speed + torch.mean(((v - target_v) / vs) ** 2)
            mask = (torch.abs(target_v) < 0.05 * vs).float()
            hold = hold + torch.sum(mask * ((v - target_v) / vs) ** 2)
            hold_n = hold_n + mask.sum()
        ddv_loss = torch.mean(((z - T[:, 1:, 1]) / zs) ** 2)
        loss = (
            position / horizon
            + 0.2 * speed / horizon
            + 0.2 * hold / (hold_n + 1)
            + 2 * ddv_loss
        )
        if not torch.isfinite(loss):
            raise ValueError("Neural training diverged")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [dw], 1)
        opt.step()
        if (step + 1) % max(1, steps // 4) == 0 or step == steps - 1:
            model = export(dw)
            score = val_rmse(model, val)
            rows.append(
                {
                    "stage": "RAM",
                    "method": "NN_joint",
                    "iteration": step + 1,
                    "validation_rmse": score,
                    "train_loss": float(loss.detach()),
                }
            )
            log(f"NN joint {step + 1}/{steps}: validation RAM RMSE={score:.6g}")
            if score < best_score:
                best_score = score
                best = copy.deepcopy(model)
                best_iteration = step + 1
    best["training"] = {
        "checkpoint": best_iteration,
        "teacher_fallback": best_iteration == 0,
        "warm_steps": warm_steps,
    }
    candidates["NN_joint"] = best
    return candidates, rows


def fit_stable(
    train,
    val,
    ddv,
    angle,
    steps=400,
    horizon=128,
    seed=42,
    hidden=12,
    log=print,
    check_cancel=None,
):
    """Learned monotone neural state model. No simulator coefficients are supplied.
    Positive nonlinear maps + negative position feedback reduce free-run instability.
    Input direction is learned; internal velocity memory is a learned convex relaxation.
    """
    if any(len(f.time) <= horizon + 34 for f in train):
        raise ValueError(
            "Every TRAIN segment must exceed horizon + 34 samples; reduce horizon."
        )
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    full = np.concatenate([f.values for f in train])
    xm = float(np.mean(full[:, 2]))
    xs = max(float(np.std(full[:, 2])), EPS)
    vs = max(
        float(np.std(np.concatenate([np.diff(f.values[:, 2]) * FS for f in train]))),
        EPS,
    )
    zs = max(float(np.std(full[:, 1])), EPS)

    class Mono(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.linspace(-1, 1, hidden))
            self.b = torch.nn.Parameter(torch.linspace(-2, 2, hidden))
            self.w2 = torch.nn.Parameter(torch.full((hidden,), -2.0))
            self.linear = torch.nn.Parameter(torch.tensor(-1.0))

        def forward(self, z):
            return torch.nn.functional.softplus(self.linear) * z + (
                torch.tanh(
                    z[..., None] * torch.nn.functional.softplus(self.w1) + self.b
                )
                * torch.nn.functional.softplus(self.w2)
            ).sum(-1)

        def export(self):
            return {
                "w1": torch.nn.functional.softplus(self.w1).detach().tolist(),
                "b": self.b.detach().tolist(),
                "w2": torch.nn.functional.softplus(self.w2).detach().tolist(),
                "linear": float(torch.nn.functional.softplus(self.linear).detach()),
            }

    g1 = Mono()
    g2 = Mono()
    direction = torch.nn.Parameter(torch.tensor(1.0))
    leak = torch.nn.Parameter(torch.tensor(-3.0))
    offset = torch.nn.Parameter(torch.tensor(0.0))
    memory = torch.nn.Parameter(torch.tensor(0.0))
    dw = torch.nn.Parameter(torch.tensor(ddv["weights"], dtype=torch.float32))
    params = (
        list(g1.parameters())
        + list(g2.parameters())
        + [direction, leak, offset, memory]
    )

    def update(z, x, v):
        target = g2(
            g1(z / zs * direction)
            - torch.nn.functional.softplus(leak) * (x - xm) / xs
            + offset
        )
        a = torch.sigmoid(memory)
        return (a * v / vs + (1 - a) * target) * vs

    arrays = []
    X = []
    Y = []
    for f in train:
        v = np.r_[0, np.diff(f.values[:, 2]) * FS]
        X.append(np.c_[f.values[1:, 1], f.values[:-1, 2], v[:-1]])
        Y.append(v[1:])
        arrays.append(
            (
                torch.tensor(
                    ddv_features(
                        f.values[:, 0],
                        ddv["scale"],
                        ddv["lags"],
                        ddv.get("basis", "legacy"),
                    ),
                    dtype=torch.float32,
                ),
                torch.tensor(f.values, dtype=torch.float32),
                torch.tensor(v, dtype=torch.float32),
            )
        )
    X = torch.tensor(np.concatenate(X), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(Y), dtype=torch.float32)
    opt = torch.optim.Adam(params, lr=0.01)
    for step in range(600):
        if check_cancel is not None:
            check_cancel()
        ix = torch.tensor(rng.integers(0, len(X), 2048))
        b = X[ix]
        loss = torch.mean(((update(b[:, 0], b[:, 1], b[:, 2]) - Y[ix]) / vs) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5)
        opt.step()

    def export():
        return {
            "ddv": dict(ddv, weights=dw.detach().tolist()),
            "angle": angle,
            "ram": {
                "kind": "monotone",
                "x_mean": xm,
                "x_scale": xs,
                "v_scale": vs,
                "ddv_scale": zs,
                "g1": g1.export(),
                "g2": g2.export(),
                "direction": float(direction.detach()),
                "leak": float(torch.nn.functional.softplus(leak).detach()),
                "offset": float(offset.detach()),
                "memory": float(torch.sigmoid(memory).detach()),
            },
        }

    best = export()
    bestscore = val_rmse(best, val)
    rows = [
        {
            "stage": "RAM",
            "method": "NN_stable",
            "iteration": 0,
            "validation_rmse": bestscore,
        }
    ]
    opt = torch.optim.Adam(
        [{"params": params, "lr": 0.002}, {"params": [dw], "lr": 0.00001}]
    )
    for step in range(steps):
        if check_cancel is not None:
            check_cancel()
        us = []
        ys = []
        v0 = []
        for b in range(24):
            features, truth, velocity = arrays[int(rng.integers(len(arrays)))]
            start = int(rng.integers(33, len(truth) - horizon))
            us.append(features[start : start + horizon])
            ys.append(truth[start - 1 : start + horizon])
            v0.append(velocity[start - 1])
        U = torch.stack(us)
        T = torch.stack(ys)
        x = T[:, 0, 2]
        v = torch.stack(v0)
        z = U @ dw
        loss = 0.0
        rate_loss = 0.0
        for k in range(horizon):
            v = update(z[:, k], x, v)
            x = x + v / FS
            target_v = (T[:, k + 1, 2] - T[:, k, 2]) * FS
            loss = loss + torch.mean(((x - T[:, k + 1, 2]) / xs) ** 2)
            rate_loss = rate_loss + torch.mean(((v - target_v) / vs) ** 2)
        loss = (loss + 0.2 * rate_loss) / horizon + 2 * torch.mean(
            ((z - T[:, 1:, 1]) / zs) ** 2
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params + [dw], 2)
        opt.step()
        if (step + 1) % max(1, steps // 4) == 0 or step == steps - 1:
            m = export()
            score = val_rmse(m, val)
            rows.append(
                {
                    "stage": "RAM",
                    "method": "NN_stable",
                    "iteration": step + 1,
                    "validation_rmse": score,
                    "train_loss": float(loss.detach()),
                }
            )
            log(f"NN stable {step + 1}/{steps}: validation RAM RMSE={score:.6g}")
            if score < bestscore:
                best = m
                bestscore = score
    return best, rows
