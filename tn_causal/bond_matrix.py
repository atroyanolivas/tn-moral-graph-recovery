# tn_causal/bond_matrix.py
import torch
import torch.nn as nn


class BondMatrix(nn.Module):
    """Factorized bond matrix B_{ij} = J_{ij} + U_{ij} V_{ij}^T.

    J baseline (all-ones) = disconnected state. C = UV^T = correction.
    Frobenius penalty is a smooth surrogate for ||C||_* (variational identity).
    """

    def __init__(self, site_i: int, site_j: int, r_max: int = 3, K: int = None):
        super().__init__()
        assert site_i < site_j, "Convention: site_i < site_j"
        self.site_i = site_i
        self.site_j = site_j
        self.r_max = r_max
        self.K = K if K is not None else r_max

        self.U = nn.Parameter(torch.empty(self.r_max, self.K))
        self.V = nn.Parameter(torch.empty(self.r_max, self.K))
        self.register_buffer("J", torch.ones(self.r_max, self.r_max))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.U, mean=0.0, std=0.01)
        nn.init.normal_(self.V, mean=0.0, std=0.01)

    def matrix(self) -> torch.Tensor:
        return self.J + self.U @ self.V.T

    def frobenius_penalty(self) -> torch.Tensor:
        return 0.5 * (torch.sum(self.U ** 2) + torch.sum(self.V ** 2))

    @torch.no_grad()
    def nuclear_norm(self) -> torch.Tensor:
        return torch.linalg.matrix_norm(self.U @ self.V.T, ord="nuc")

    @torch.no_grad()
    def effective_rank(self, tol: float = 1e-6) -> int:
        s = torch.linalg.svdvals(self.U @ self.V.T)
        return int((s > tol).sum().item())

    def extra_repr(self) -> str:
        return f"sites=({self.site_i},{self.site_j}), r_max={self.r_max}, K={self.K}"


class DirectBondMatrix(nn.Module):
    """Direct bond matrix: B = J + C. Penalize ||C||_* via SVD directly."""

    def __init__(self, site_i: int, site_j: int, r_max: int = 3, K: int = None):
        super().__init__()
        assert site_i < site_j
        self.site_i = site_i
        self.site_j = site_j
        self.r_max = r_max
        self.K = K if K is not None else r_max

        self.C = nn.Parameter(torch.zeros(self.r_max, self.r_max))
        self.register_buffer("J", torch.ones(self.r_max, self.r_max))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.C, mean=0.0, std=0.01)

    def matrix(self) -> torch.Tensor:
        return self.J + self.C

    def penalty(self) -> torch.Tensor:
        return torch.linalg.svdvals(self.C).sum()

    def frobenius_penalty(self) -> torch.Tensor:
        return self.penalty()

    @torch.no_grad()
    def nuclear_norm(self) -> torch.Tensor:
        return torch.linalg.svdvals(self.C).sum()

    @torch.no_grad()
    def effective_rank(self, tol: float = 1e-6) -> int:
        s = torch.linalg.svdvals(self.C)
        return int((s > tol).sum().item())

    def extra_repr(self) -> str:
        return f"sites=({self.site_i},{self.site_j}), r_max={self.r_max}, direct=True"