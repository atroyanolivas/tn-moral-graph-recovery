import torch
import pytest
from tn_causal.fctn import FCTN
from typing import Optional


def select_tau(
    nuclear_norms: dict[tuple[int, int], float],
    min_gap_ratio: float = 2.0,
) -> Optional[float]:
    """Select threshold via maximum gap ratio in sorted nuclear norms.

    Sorts norms descending. Finds the consecutive pair with the largest
    ratio exceeding min_gap_ratio. Returns the geometric mean of that
    pair as tau. Returns None if no gap exceeds min_gap_ratio (keep all).

    The geometric mean sits at the center of the gap in log-space,
    maximizing the margin to both clusters.
    """
    sorted_norms = sorted(nuclear_norms.values(), reverse=True)
    if len(sorted_norms) <= 1:
        return None

    best_ratio = min_gap_ratio
    best_gap = None

    for k in range(len(sorted_norms) - 1):
        upper = sorted_norms[k]
        lower = sorted_norms[k + 1]
        if lower > 1e-12:
            ratio = upper / lower
            if ratio > best_ratio:
                best_ratio = ratio
                best_gap = (upper, lower)

    if best_gap is None:
        return None

    return (best_gap[0] * best_gap[1]) ** 0.5


def recover_graph(
    nuclear_norms: dict[tuple[int, int], float],
    tau: Optional[float],
) -> list[tuple[int, int]]:
    """Recover effective graph by thresholding nuclear norms."""
    if tau is None:
        return list(nuclear_norms.keys())
    return [e for e, norm in nuclear_norms.items() if norm > tau]



# ____________________ Tests ________________________

def test_tau_fork_not_rank_faithful():
    """Fork with full-rank conditionals (d=3, r=3).

    Rank-based recovery FAILS: all bonds have deviation rank >= 1.
    Nuclear norm gap correctly identifies X-Y as non-moral.

    Moral graph: {X-Z, Y-Z} = {(0,2), (1,2)}
    """
    torch.manual_seed(42)
    d, r, K = 3, 3, 3

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
    p = p / p.sum()

    fctn = FCTN(dims=[d, d, d], r_max=r, K=K)
    with torch.no_grad():
        for bond in fctn.bonds.values():
            W = torch.randn(bond.r_max, bond.K) * 0.5
            bond.U.copy_(W)
            bond.V.copy_(W)

    beta = 0.0005
    n_steps = 3000
    optimizer = torch.optim.Adam(fctn.parameters(), lr=0.01)

    for i in range(n_steps):
        optimizer.zero_grad()
        loss = fctn.loss(p, beta)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()
        norms = {k: v.item() for k, v in fctn.bond_nuclear_norms().items()}

    tau = select_tau(norms)
    recovered = set(recover_graph(norms, tau))
    expected = {(0, 2), (1, 2)}

    sorted_norms = sorted(norms.values(), reverse=True)
    ratios = [sorted_norms[k] / sorted_norms[k+1] for k in range(len(sorted_norms)-1) if sorted_norms[k+1] > 1e-12]

    print(f"\n  Reconstruction error: {recon:.2e}")
    print(f"  Bond norms: { {k: f'{v:.4f}' for k, v in norms.items()} }")
    print(f"  Sorted norms: {[round(n, 4) for n in sorted_norms]}")
    print(f"  Gap ratios: {[round(r, 2) for r in ratios]}")
    print(f"  Selected tau: {tau}")
    print(f"  Recovered: {recovered}  Expected: {expected}")

    assert recon < 1e-3, f"Reconstruction error too high: {recon:.2e}"
    assert tau is not None, f"No gap found. Norms: {norms}"
    assert recovered == expected, f"Wrong graph: {recovered} != {expected}"


def test_tau_collider():
    """Collider DAG: X -> Z <- Y. Moral graph is a triangle (all edges).

    All 3 bonds carry irreducible information. No gap should exceed
    min_gap_ratio → tau = None → keep all edges.

    Moral graph: {X-Y, X-Z, Y-Z} = {(0,1), (0,2), (1,2)}
    """
    torch.manual_seed(42)
    d, r, K = 5, 3, 3

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
    p = p / p.sum()

    fctn = FCTN(dims=[d, d, d], r_max=r, K=K)
    with torch.no_grad():
        for bond in fctn.bonds.values():
            W = torch.randn(bond.r_max, bond.K) * 0.5
            bond.U.copy_(W)
            bond.V.copy_(W)

    beta = 0.0005
    n_steps = 3000
    optimizer = torch.optim.Adam(fctn.parameters(), lr=0.01)

    for i in range(n_steps):
        optimizer.zero_grad()
        loss = fctn.loss(p, beta)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()
        norms = {k: v.item() for k, v in fctn.bond_nuclear_norms().items()}

    tau = select_tau(norms)
    recovered = set(recover_graph(norms, tau))
    expected = {(0, 1), (0, 2), (1, 2)}

    sorted_norms = sorted(norms.values(), reverse=True)
    ratios = [sorted_norms[k] / sorted_norms[k+1] for k in range(len(sorted_norms)-1) if sorted_norms[k+1] > 1e-12]

    print(f"\n  Reconstruction error: {recon:.2e}")
    print(f"  Bond norms: { {k: f'{v:.4f}' for k, v in norms.items()} }")
    print(f"  Sorted norms: {[round(n, 4) for n in sorted_norms]}")
    print(f"  Gap ratios: {[round(r, 2) for r in ratios]}")
    print(f"  Selected tau: {tau}")
    print(f"  Recovered: {recovered}  Expected: {expected}")

    assert recon < 1e-3, f"Reconstruction error too high: {recon:.2e}"
    assert recovered == expected, f"Wrong graph: {recovered} != {expected}"


def test_tau_single_edge():
    """Z independent of X and Y; only X-Y dependent.
    Two bonds should be pruned, one kept.

    Moral graph: {X-Y} = {(0,1)}
    """
    torch.manual_seed(42)
    d, r, K = 5, 3, 3

    p_x = torch.tensor([0.25, 0.2, 0.2, 0.2, 0.15])
    p_z = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1])

    q_y = torch.tensor([
        [0.5, 0.3, 0.1, 0.05, 0.05],
        [0.05, 0.1, 0.3, 0.35, 0.2],
    ])
    w_y = torch.softmax(torch.tensor([
        [2.0, -1.0],
        [1.0,  0.0],
        [0.0,  0.5],
        [-1.0, 1.0],
        [-2.0, 2.0],
    ]), dim=1)
    p_y_given_x = w_y @ q_y

    p = torch.zeros(d, d, d)
    for x in range(d):
        for y in range(d):
            for z in range(d):
                p[x, y, z] = p_x[x] * p_y_given_x[x, y] * p_z[z]
    p = p / p.sum()

    fctn = FCTN(dims=[d, d, d], r_max=r, K=K)
    with torch.no_grad():
        for bond in fctn.bonds.values():
            W = torch.randn(bond.r_max, bond.K) * 0.5
            bond.U.copy_(W)
            bond.V.copy_(W)

    beta = 0.0005
    n_steps = 3000
    optimizer = torch.optim.Adam(fctn.parameters(), lr=0.01)

    for i in range(n_steps):
        optimizer.zero_grad()
        loss = fctn.loss(p, beta)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()
        norms = {k: v.item() for k, v in fctn.bond_nuclear_norms().items()}

    tau = select_tau(norms)
    recovered = set(recover_graph(norms, tau))
    expected = {(0, 1)}

    sorted_norms = sorted(norms.values(), reverse=True)
    ratios = [sorted_norms[k] / sorted_norms[k+1] for k in range(len(sorted_norms)-1) if sorted_norms[k+1] > 1e-12]

    print(f"\n  Reconstruction error: {recon:.2e}")
    print(f"  Bond norms: { {k: f'{v:.4f}' for k, v in norms.items()} }")
    print(f"  Sorted norms: {[round(n, 4) for n in sorted_norms]}")
    print(f"  Gap ratios: {[round(r, 2) for r in ratios]}")
    print(f"  Selected tau: {tau}")
    print(f"  Recovered: {recovered}  Expected: {expected}")

    assert recon < 1e-3, f"Reconstruction error too high: {recon:.2e}"
    assert tau is not None, f"No gap found. Norms: {norms}"
    assert recovered == expected, f"Wrong graph: {recovered} != {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])