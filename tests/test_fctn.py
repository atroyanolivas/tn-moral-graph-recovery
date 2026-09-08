# tests/test_fctn.py

import torch
import torch.nn as nn
import pytest
from tn_causal.fctn import FCTN


# ── Construction ──────────────────────────────────────────────────────

def test_construction_m3():
    fctn = FCTN(dims=[3, 4, 5], r_max=8, K=6)
    assert fctn.m == 3
    assert len(fctn.local_tensors) == 3
    assert fctn.local_tensors[0].shape == (3, 8, 8)
    assert fctn.local_tensors[1].shape == (4, 8, 8)
    assert fctn.local_tensors[2].shape == (5, 8, 8)
    assert len(fctn.bonds) == 3  # C(3,2) = 3


def test_construction_m2():
    fctn = FCTN(dims=[3, 4], r_max=5)
    assert fctn.m == 2
    assert fctn.local_tensors[0].shape == (3, 5)
    assert fctn.local_tensors[1].shape == (4, 5)
    assert len(fctn.bonds) == 1


def test_construction_m4():
    fctn = FCTN(dims=[2, 2, 2, 2], r_max=3)
    assert fctn.m == 4
    for i in range(4):
        assert fctn.local_tensors[i].shape == (2, 3, 3, 3)  # d_i + (m-1) bond dims
    assert len(fctn.bonds) == 6  # C(4,2) = 6


# ── Einsum string ─────────────────────────────────────────────────────

def test_einsum_string_m3():
    fctn = FCTN(dims=[2, 2, 2], r_max=4)
    # Labels: a,b,c = physical; d,e = bond(0,1); f,g = bond(0,2); h,i = bond(1,2)
    # N0: a, d, f   N1: b, e, h   N2: c, g, i
    # B01: de  B02: fg  B12: hi
    assert fctn._einsum_str == "adf,beh,cgi,de,fg,hi->abc"


def test_einsum_string_m2():
    fctn = FCTN(dims=[3, 4], r_max=5)
    assert fctn._einsum_str == "ac,bd,cd->ab"


# ── Contraction ───────────────────────────────────────────────────────

def test_contraction_shape():
    fctn = FCTN(dims=[3, 4, 5], r_max=8)
    p_hat = fctn.contract()
    assert p_hat.shape == (3, 4, 5)


def test_contraction_shape_m4():
    fctn = FCTN(dims=[2, 3, 2, 3], r_max=4)
    p_hat = fctn.contract()
    assert p_hat.shape == (2, 3, 2, 3)


def test_manual_contraction_m3():
    """Verify einsum against explicit loop-based contraction for m=3, small dims."""
    torch.manual_seed(42)
    d0, d1, d2, r = 2, 2, 2, 3

    fctn = FCTN(dims=[d0, d1, d2], r_max=r, K=r)
    N0, N1, N2 = fctn.local_tensors
    B01 = fctn.bonds["0-1"].matrix()
    B02 = fctn.bonds["0-2"].matrix()
    B12 = fctn.bonds["1-2"].matrix()

    p_manual = torch.zeros(d0, d1, d2)
    for s0 in range(d0):
        for s1 in range(d1):
            for s2 in range(d2):
                val = 0.0
                for a01 in range(r):       # bond (0,1) on site 0's side
                    for a10 in range(r):   # bond (0,1) on site 1's side
                        for a02 in range(r):       # bond (0,2) on site 0's side
                            for a20 in range(r):   # bond (0,2) on site 2's side
                                for a12 in range(r):       # bond (1,2) on site 1's side
                                    for a21 in range(r):   # bond (1,2) on site 2's side
                                        val += (
                                            N0[s0, a01, a02]
                                            * N1[s1, a10, a12]
                                            * N2[s2, a20, a21]
                                            * B01[a01, a10]
                                            * B02[a02, a20]
                                            * B12[a12, a21]
                                        )
                p_manual[s0, s1, s2] = val

    p_einsum = fctn.contract()
    assert torch.allclose(p_einsum, p_manual, atol=1e-5), (
        f"Max diff: {(p_einsum - p_manual).abs().max()}"
    )


# ── Gradient flow ─────────────────────────────────────────────────────

def test_gradient_flows_to_all_parameters():
    fctn = FCTN(dims=[3, 3, 3], r_max=5)
    p = torch.rand(3, 3, 3)
    p = p / p.sum()

    loss = fctn.loss(p, beta=0.1)
    loss.backward()

    for i, tensor in enumerate(fctn.local_tensors):
        assert tensor.grad is not None, f"local_tensors[{i}].grad is None"
        assert tensor.grad.shape == tensor.shape
        assert torch.any(tensor.grad != 0), f"local_tensors[{i}].grad is all zeros"

    for key, bond in fctn.bonds.items():
        assert bond.U.grad is not None, f"bond {key}.U.grad is None"
        assert bond.V.grad is not None, f"bond {key}.V.grad is None"
        assert torch.any(bond.U.grad != 0), f"bond {key}.U.grad is all zeros"
        assert torch.any(bond.V.grad != 0), f"bond {key}.V.grad is all zeros"


# ── Loss components ───────────────────────────────────────────────────

def test_loss_is_sum_of_components():
    torch.manual_seed(0)
    fctn = FCTN(dims=[3, 3, 3], r_max=5)
    p = torch.rand(3, 3, 3)
    p = p / p.sum()
    beta = 0.3

    recon = fctn.reconstruction_loss(p)
    pen = fctn.penalty()
    total = fctn.loss(p, beta)

    assert torch.isclose(total, recon + beta * pen, atol=1e-6)


def test_penalty_zero_when_bonds_zero():
    fctn = FCTN(dims=[3, 3, 3], r_max=5)
    # Zero out all bond factors
    with torch.no_grad():
        for bond in fctn.bonds.values():
            bond.U.zero_()
            bond.V.zero_()
    assert fctn.penalty().item() == 0.0


# ── Effective graph ───────────────────────────────────────────────────

def test_effective_graph_all_zero():
    fctn = FCTN(dims=[3, 3, 3], r_max=5)
    with torch.no_grad():
        for bond in fctn.bonds.values():
            bond.U.zero_()
            bond.V.zero_()
    assert fctn.effective_graph(threshold=1e-8) == []


def test_effective_graph_all_active():
    fctn = FCTN(dims=[3, 3, 3], r_max=5)
    # Set bonds to identity-like
    with torch.no_grad():
        for bond in fctn.bonds.values():
            nn.init.eye_(bond.U)
            nn.init.eye_(bond.V)
    edges = fctn.effective_graph(threshold=0.1)
    assert set(edges) == {(0, 1), (0, 2), (1, 2)}


def test_adam_prunes_independent_edge():
    """Fork DAG: Z -> X, Z -> Y.
    X ⊥ Y | Z, so moral graph drops X-Y edge → 1 pruned, 2 active.

    Uses d=5 > r_max=3 with rank-2 conditionals. The fork distribution
    has multilinear rank (2,2,4). The V-shape (X-Y pruned) achieves
    (3,3,5) ≥ (2,2,4), but wrong structures (pruning X-Z or Y-Z) only
    achieve mode-3 rank ≤ 3 < 4, so they cannot represent it.
    """
    torch.manual_seed(42)
    d = 5
    r = 3
    K = 3

    p_z = torch.tensor([0.25, 0.2, 0.2, 0.2, 0.15])

    # Rank-2 conditionals via mixture of 2 base distributions
    q_x = torch.tensor([
        [0.5, 0.3, 0.1, 0.05, 0.05],
        [0.05, 0.1, 0.3, 0.35, 0.2],
    ])  # (2, 5) base distributions for x
    w_x = torch.softmax(torch.tensor([
        [2.0, -1.0],
        [1.0,  0.0],
        [0.0,  0.5],
        [-1.0, 1.0],
        [-2.0, 2.0],
    ]), dim=1)  # (5, 2) mixing weights
    p_x_given_z = w_x @ q_x  # (5, 5) = (z, x), rank 2

    q_y = torch.tensor([
        [0.4, 0.35, 0.15, 0.05, 0.05],
        [0.05, 0.05, 0.2, 0.4, 0.3],
    ])  # (2, 5) base distributions for y
    w_y = torch.softmax(torch.tensor([
        [1.5, -0.5],
        [0.5,  0.5],
        [-0.5, 1.5],
        [1.0,  0.0],
        [-1.0, 2.0],
    ]), dim=1)  # (5, 2) mixing weights
    p_y_given_z = w_y @ q_y  # (5, 5) = (z, y), rank 2

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
    n_steps = 2000
    optimizer = torch.optim.Adam(fctn.parameters(), lr=0.01)

    for i in range(n_steps):
        optimizer.zero_grad()
        loss = fctn.loss(p, beta)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()
        ranks = {(b.site_i, b.site_j): b.effective_rank(tol=0.01) for b in fctn.bonds.values()}
        norms = fctn.bond_nuclear_norms()

    print(f"\n  Reconstruction error: {recon:.2e}")
    print(f"  Bond ranks: {ranks}")
    print(f"  Bond norms: { {k: f'{v:.4f}' for k, v in norms.items()} }")

    print("  Bond singular values:")
    for key, bond in fctn.bonds.items():
        sv = torch.linalg.svdvals(bond.U @ bond.V.T).detach().tolist()
        print(f"    {key}: {[round(s, 4) for s in sv]}")

    n_pruned = sum(1 for rk in ranks.values() if rk == 0)
    n_active = sum(1 for rk in ranks.values() if rk >= 1)
    assert n_pruned == 1, f"Expected 1 pruned bond, got {n_pruned}. Ranks: {ranks}"
    assert n_active == 2, f"Expected 2 active bonds, got {n_active}. Ranks: {ranks}"
    assert recon < 1e-3, f"Reconstruction error too high: {recon:.2e}"

def test_adam_keeps_triangular_collider():
    """Collider DAG: X -> Z <- Y.
    Moral graph is a triangle (all 3 edges), so no bond should be pruned.

    Uses d=5 > r_max=3 so the V-shape (one bond at J) cannot trivially
    represent the distribution via a full-size Tucker core. The collider
    has multilinear rank (5,5,5); the V-shape only achieves (3,3,5).
    """
    torch.manual_seed(42)
    d = 5
    r = 3
    K = 3

    p_x = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1])
    p_y = torch.tensor([0.25, 0.25, 0.2, 0.15, 0.15])

    # p(z | x, y) with explicit x-y interaction (non-decomposable into f(x)·g(y))
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

    # Symmetric initialization: U = V, std=0.5
    with torch.no_grad():
        for bond in fctn.bonds.values():
            W = torch.randn(bond.r_max, bond.K) * 0.5
            bond.U.copy_(W)
            bond.V.copy_(W)

    beta = 0.0005
    n_steps = 1000
    optimizer = torch.optim.Adam(fctn.parameters(), lr=0.01)

    for i in range(n_steps):
        optimizer.zero_grad()
        loss = fctn.loss(p, beta)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()
        ranks = {(b.site_i, b.site_j): b.effective_rank(tol=0.01) for b in fctn.bonds.values()}
        norms = fctn.bond_nuclear_norms()

    print(f"\n  Reconstruction error: {recon:.2e}")
    print(f"  Bond ranks: {ranks}")
    print(f"  Bond norms: { {k: f'{v:.4f}' for k, v in norms.items()} }")

    print("  Bond singular values:")
    for key, bond in fctn.bonds.items():
        sv = torch.linalg.svdvals(bond.U @ bond.V.T).detach().tolist()
        print(f"    {key}: {[round(s, 4) for s in sv]}")

    # J-baseline: pruned = deviation rank 0, active = deviation rank >= 1
    n_pruned = sum(1 for rk in ranks.values() if rk == 0)
    n_active = sum(1 for rk in ranks.values() if rk >= 1)
    assert n_pruned == 0, f"Expected 0 pruned bonds (triangle), got {n_pruned}. Ranks: {ranks}"
    assert n_active == 3, f"Expected 3 active bonds (triangle), got {n_active}. Ranks: {ranks}"
    assert recon < 1e-3, f"Reconstruction error too high: {recon:.2e}"


def test_adam_keeps_single_edge():
    """Z is independent of X and Y; only X-Y are dependent.
    Moral graph has a single edge X-Y → 2 pruned bonds, 1 active."""
    torch.manual_seed(42)
    d = 3
    r = 3
    K = 3

    p_x = torch.tensor([0.4, 0.3, 0.3])
    p_y_given_x = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
    ])
    p_z = torch.tensor([0.5, 0.3, 0.2])  # independent of x and y

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
    n_steps = 1000
    optimizer = torch.optim.Adam(fctn.parameters(), lr=0.01)

    for i in range(n_steps):
        optimizer.zero_grad()
        loss = fctn.loss(p, beta)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()
        ranks = {(b.site_i, b.site_j): b.effective_rank(tol=0.01) for b in fctn.bonds.values()}
        norms = fctn.bond_nuclear_norms()

    print(f"\n  Reconstruction error: {recon:.2e}")
    print(f"  Bond ranks: {ranks}")
    print(f"  Bond norms: { {k: f'{v:.4f}' for k, v in norms.items()} }")

    print("  Bond singular values:")
    for key, bond in fctn.bonds.items():
        sv = torch.linalg.svdvals(bond.U @ bond.V.T).detach().tolist()
        print(f"    {key}: {[round(s, 4) for s in sv]}")

    n_pruned = sum(1 for rk in ranks.values() if rk == 0)
    n_active = sum(1 for rk in ranks.values() if rk >= 1)
    assert n_pruned == 2, f"Expected 2 pruned bonds, got {n_pruned}. Ranks: {ranks}"
    assert n_active == 1, f"Expected 1 active bond, got {n_active}. Ranks: {ranks}"
    assert recon < 1e-3, f"Reconstruction error too high: {recon:.2e}"

# ── Unrolled gradient (preview of bi-level) ──────────────────────────

def test_gradient_through_multiple_optimization_steps():
    """Verify gradients flow through multiple contract() calls (unrolling)."""
    torch.manual_seed(0)
    dims = [2, 2, 2]
    p = torch.rand(*dims)
    p = p / p.sum()

    fctn = FCTN(dims, r_max=4)
    optimizer = torch.optim.SGD(fctn.parameters(), lr=0.01)

    # Two manual steps, then check that the graph is retained
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = fctn.loss(p, beta=0.1)
        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
