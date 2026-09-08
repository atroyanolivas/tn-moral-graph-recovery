"""
Synthetic moral-graph recovery experiments.

Run from the project root:
    python examples/synthetic_experiments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tn_causal.fctn import FCTN
from tn_causal.optimizer import GradientInnerOptimizer
from tn_causal.recovery import recover_graph_by_cmi


# --------------------------------------------------------------------- #
#  Data generators                                                       #
# --------------------------------------------------------------------- #

def make_chain(d: int = 4) -> torch.Tensor:
    """Chain: X -> Z -> Y. Moral graph: {(0,2), (1,2)} (X=0, Y=1, Z=2)."""
    p_x = torch.tensor([0.30, 0.25, 0.25, 0.20])

    p_z_given_x = torch.zeros(d, d)
    for x in range(d):
        logits = torch.zeros(d)
        for z in range(d):
            logits[z] = 2.5 * (x == z) + 0.5
        p_z_given_x[x] = torch.softmax(logits, dim=0)

    p_y_given_z = torch.zeros(d, d)
    for z in range(d):
        logits = torch.zeros(d)
        for y in range(d):
            logits[y] = 2.0 * (z == y) + 0.3 * ((z + 1) % d == y)
        p_y_given_z[z] = torch.softmax(logits, dim=0)

    p = torch.zeros(d, d, d)
    for x in range(d):
        for z in range(d):
            for y in range(d):
                p[x, y, z] = p_x[x] * p_z_given_x[x, z] * p_y_given_z[z, y]
    return p / p.sum()


def make_fork(d: int = 3) -> torch.Tensor:
    """Fork: Z -> X, Z -> Y. Moral graph: {(0,2), (1,2)} (X=0, Y=1, Z=2)."""
    p_z = torch.tensor([0.4, 0.3, 0.3])
    p_x_given_z = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
    ])
    p_y_given_z = torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
        [0.3, 0.3, 0.4],
    ])
    p = torch.zeros(d, d, d)
    for z in range(d):
        for x in range(d):
            for y in range(d):
                p[x, y, z] = p_z[z] * p_x_given_z[z, x] * p_y_given_z[z, y]
    return p / p.sum()


def make_collider(d: int = 5) -> torch.Tensor:
    """Collider: X -> Z <- Y. Moral graph: triangle, {(0,1), (0,2), (1,2)}."""
    p_x = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1])
    p_y = torch.tensor([0.25, 0.25, 0.2, 0.15, 0.15])

    p_z_given_xy = torch.zeros(d, d, d)
    for x in range(d):
        for y in range(d):
            logits = torch.zeros(d)
            for z in range(d):
                logits[z] = 2.0 * (x == z) + 2.0 * (y == z) + 1.5 * (x == y and z == x)
            p_z_given_xy[x, y] = torch.softmax(logits, dim=0)

    p = torch.zeros(d, d, d)
    for x in range(d):
        for y in range(d):
            for z in range(d):
                p[x, y, z] = p_x[x] * p_y[y] * p_z_given_xy[x, y, z]
    return p / p.sum()


def make_diamond(d: int = 3) -> torch.Tensor:
    """DAG: X -> Z, Y -> Z, Z -> W.

    Variables: (0,1,2,3) = (X, Y, Z, W).
    Moral graph: (0,1) added by v-structure at Z, plus (0,2), (1,2), (2,3):
        {(0,1), (0,2), (1,2), (2,3)}
    """
    p_x = torch.full((d,), 1.0 / d)
    p_y = torch.full((d,), 1.0 / d)

    p_z_given_xy = torch.zeros(d, d, d)
    for x in range(d):
        for y in range(d):
            logits = torch.zeros(d)
            for z in range(d):
                logits[z] = 2.0 * (x == z) + 1.5 * (y == z) + 0.2
            p_z_given_xy[x, y] = torch.softmax(logits, dim=0)

    p_w_given_z = torch.zeros(d, d)
    for z in range(d):
        logits = torch.zeros(d)
        for w in range(d):
            logits[w] = 2.0 * (w == z) + 0.3 * ((z + 1) % d == w)
        p_w_given_z[z] = torch.softmax(logits, dim=0)

    p = torch.zeros(d, d, d, d)
    for x in range(d):
        for y in range(d):
            for z in range(d):
                for w in range(d):
                    p[x, y, z, w] = p_x[x] * p_y[y] * p_z_given_xy[x, y, z] * p_w_given_z[z, w]
    return p / p.sum()


# --------------------------------------------------------------------- #
#  Setup / runner (same pattern as your existing recover_moral_graph.py) #
# --------------------------------------------------------------------- #

def init_bonds(fctn: FCTN) -> None:
    """Initialize bonds with U = V (symmetric start)."""
    with torch.no_grad():
        for bond in fctn.bonds.values():
            W = torch.randn(bond.r_max, bond.K) * 0.5
            bond.U.copy_(W)
            bond.V.copy_(W)


def run_recovery(
    name: str,
    p: torch.Tensor,
    expected: set[tuple[int, int]],
    r_max: int,
    beta: float,
    lr: float = 0.01,
    max_iter: int = 3000,
    seed: int = 42,
) -> dict:
    """Fit an FCTN to p, extract G_eff, compare with the moral graph."""
    m = p.dim()
    d = p.shape[0]

    print(f"\n{'=' * 74}")
    print(f"  {name[:72]}")
    print(f"  m={m}, d={d}, r_max={r_max}, beta={beta:.2e}, max_iter={max_iter}")
    print(f"  Expected moral graph: {sorted(expected)}")
    print("=" * 74)

    torch.manual_seed(seed)
    fctn = FCTN(dims=[d] * m, r_max=r_max, K=r_max)
    init_bonds(fctn)

    optimizer = GradientInnerOptimizer(lr=lr)
    optimizer.optimize(fctn, p, beta, max_iter=max_iter, tol=0.0)

    with torch.no_grad():
        eps = fctn.reconstruction_loss(p).item()
        p_hat = fctn.contract().clamp(min=0.0)
        p_hat = p_hat / p_hat.sum()

    recovered = set(fctn.effective_graph())

    # Post-hoc I_C verification (the paper's recovery.py diagnostic)
    cmi_recovered = set(recover_graph_by_cmi(p_hat, tol=1e-3))

    tp = len(recovered & expected)
    fp = len(recovered - expected)
    fn = len(expected - recovered)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  Reconstruction error    eps* = {eps:.3e}")
    print(f"  Recovered G_eff (rank)  : {sorted(recovered)}")
    print(f"  Recovered G_eff (CMI)   : {sorted(cmi_recovered)}")
    print(f"  Expected G^m            : {sorted(expected)}")
    print(f"  Exact?                  : {recovered == expected}")
    print(f"  Precision / Recall / F1 : {precision:.3f} / {recall:.3f} / {f1:.3f}")

    return {
        "name": name,
        "m": m,
        "d": d,
        "n_edges": len(expected),
        "beta": beta,
        "eps": eps,
        "recovered": recovered,
        "expected": expected,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "match": recovered == expected,
    }


def main() -> None:
    print("Tensor Network Causal Discovery — Synthetic Experiments")
    print("Moral graph recovery via FCTN + nuclear-norm regularization\n")

    cases = [
        ("1. Chain   (X -> Z -> Y)",            make_chain(d=4),   {(0, 2), (1, 2)},         2, 1e-3),
        ("2. Fork    (Z -> X, Z -> Y)",         make_fork(d=3),    {(0, 2), (1, 2)},         3, 1e-3),
        ("3. Collider (X -> Z <- Y)",           make_collider(d=5),{(0, 1), (0, 2), (1, 2)}, 3, 1e-4),
        ("4. Diamond (X->Z<-Y, Z->W)",          make_diamond(d=4), {(0, 1), (0, 2), (1, 2), (2, 3)}, 3, 1.7e-2),
    ]

    results = []
    for name, p, expected, r_max, beta in cases:
        results.append(run_recovery(name, p, expected, r_max=r_max, beta=beta))

    print("\n\n" + "=" * 74)
    print("SUMMARY (copy into the paper table)")
    print("=" * 74)
    header = f"| {'Model':<28} | {'m':>2} | {'d':>2} | {'|E^m|':>4} | {'beta':>8} | {'eps*':>9} | {'Prec':>5} | {'Rec':>5} | {'F1':>5} | {'Exact':>5} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for r in results:
        print(f"| {r['name']:<20} | {r['m']:>2} | {r['d']:>2} | {r['n_edges']:>6} | "
              f"{r['beta']:.1e} | {r['eps']:.1e} | {r['precision']:.3f} | "
              f"{r['recall']:.3f} | {r['f1']:.3f} | {'YES' if r['match'] else 'no':>5} |")


if __name__ == "__main__":
    main()
