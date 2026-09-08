# tests/test_bond.py

import torch
from tn_causal.bond_matrix import BondMatrix


def test_bond_shapes():
    bond = BondMatrix(site_i=0, site_j=1, r_max=8, K=8)
    assert bond.matrix().shape == (8, 8)
    assert bond.frobenius_penalty().shape == ()  # scalar


def test_bond_gradient_flows():
    bond = BondMatrix(0, 1, r_max=8, K=8)
    loss = bond.frobenius_penalty()
    loss.backward()
    assert bond.U.grad is not None
    assert bond.V.grad is not None
    assert bond.U.grad.shape == bond.U.shape


def test_zero_init_gives_near_zero_matrix():
    bond = BondMatrix(0, 1, r_max=8, K=8)
    assert bond.matrix().abs().max() < 0.5  # small but not exactly zero


def test_nuclear_norm_no_grad():
    bond = BondMatrix(0, 1, r_max=8, K=8)
    nn_val = bond.nuclear_norm()
    assert not nn_val.requires_grad


if __name__ == "__main__":
    test_bond_shapes()
    test_bond_gradient_flows()
    test_zero_init_gives_near_zero_matrix()
    test_nuclear_norm_no_grad()
    print("All bond tests passed.")