# test_optimizer.py
import torch
import pytest
from tn_causal.fctn import FCTN
from tn_causal.optimizer import GradientInnerOptimizer


def get_fork_distribution(d=3):
    """Helper to create a simple 3-variable distribution."""
    torch.manual_seed(42)
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


def init_fctn_bonds(fctn: FCTN):
    """Initialize bonds exactly as done in test_thresholding.py"""
    with torch.no_grad():
        for bond in fctn.bonds.values():
            W = torch.randn(bond.r_max, bond.K) * 0.5
            bond.U.copy_(W)
            bond.V.copy_(W)


def test_optimizer_basic_convergence():
    """Ensure the optimizer runs full iterations and reduces reconstruction error."""
    torch.manual_seed(42)
    p = get_fork_distribution()
    fctn = FCTN(dims=[3, 3, 3], r_max=3, K=3)
    init_fctn_bonds(fctn)

    beta = 0.0005
    optimizer = GradientInnerOptimizer(lr=0.01)
    
    # tol=0.0 forces it to run all max_iter steps
    history = optimizer.optimize(fctn, p, beta, max_iter=3000, tol=0.0)

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()

    assert recon < 1e-3, f"Reconstruction error too high: {recon:.2e}"
    assert len(history["loss"]) == 3000, "Should have run all iterations"


def test_optimizer_early_stopping():
    """Verify that early stopping triggers when recon < tol."""
    torch.manual_seed(42)
    p = get_fork_distribution()
    fctn = FCTN(dims=[3, 3, 3], r_max=3, K=3)
    init_fctn_bonds(fctn)

    beta = 0.0005
    optimizer = GradientInnerOptimizer(lr=0.01)
    
    # Set a tolerance we know it can achieve
    history = optimizer.optimize(fctn, p, beta, max_iter=3000, tol=1e-4)

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()

    assert recon < 1e-4, f"Did not reach tolerance: {recon:.2e}"
    assert len(history["loss"]) < 3000, "Should have early stopped"


def test_optimizer_step_by_step():
    """Test using step() directly and verify optimizer state is maintained."""
    torch.manual_seed(42)
    p = get_fork_distribution()
    fctn = FCTN(dims=[3, 3, 3], r_max=3, K=3)
    init_fctn_bonds(fctn)

    beta = 0.0005
    optimizer = GradientInnerOptimizer(lr=0.01)
    optimizer.reset()

    # Compute initial loss
    initial_loss = fctn.loss(p, beta).item()

    # Run 100 steps manually
    for _ in range(100):
        loss = optimizer.step(fctn, p, beta)

    final_loss = loss.item()
    
    assert final_loss < initial_loss, "Loss should decrease after 100 steps"
    # Verify the internal optimizer was created and holds state (e.g., Adam moments)
    assert optimizer._optimizer is not None, "Internal optimizer should be lazily created"
    assert len(optimizer._optimizer.state) > 0, "Optimizer should have state for parameters"


def test_optimizer_lbfgs():
    """Ensure the L-BFGS closure path works without errors."""
    torch.manual_seed(42)
    p = get_fork_distribution()
    fctn = FCTN(dims=[3, 3, 3], r_max=3, K=3)
    init_fctn_bonds(fctn)

    beta = 0.0005
    optimizer = GradientInnerOptimizer(
        lr=0.1, 
        optimizer_cls=torch.optim.LBFGS,
        max_iter=20, # L-BFGS specific kwarg passed to opt_kwargs
    )
    
    history = optimizer.optimize(fctn, p, beta, max_iter=50, tol=1e-4)

    with torch.no_grad():
        recon = fctn.reconstruction_loss(p).item()

    assert recon < 1e-3, f"L-BFGS Reconstruction error too high: {recon:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])