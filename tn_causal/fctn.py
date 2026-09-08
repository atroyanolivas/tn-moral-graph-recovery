# tn_causal/fctn.py
import string
import torch
import torch.nn as nn

from .bond_matrix import BondMatrix


class FCTN(nn.Module):
    """Fully Connected Tensor Network with factorized bond matrices.

    All C(m,2) edges present, each with B_{ij} = J + C_{ij}.
    The nuclear norm penalty drives unnecessary edges to C=0 during optimization.
    """

    def __init__(self, dims: list[int], r_max: int, K: int = None, bond_cls=BondMatrix):
        super().__init__()
        assert len(dims) >= 2, "FCTN requires at least 2 sites"
        assert r_max >= 1, "r_max must be a positive integer"

        self.dims = list(dims)
        self.m = len(dims)
        self.r_max = r_max
        self.K = K if K is not None else r_max

        # Local tensors: N^{[i]} has shape (d_i, r_max, ..., r_max) with m-1 bond dims
        self.local_tensors = nn.ParameterList()
        for i in range(self.m):
            shape = (self.dims[i],) + (self.r_max,) * (self.m - 1)
            self.local_tensors.append(nn.Parameter(torch.empty(*shape)))

        # Bond matrices: one per edge {i,j} with i < j
        self.bonds = nn.ModuleDict()
        for i in range(self.m):
            for j in range(i + 1, self.m):
                self.bonds[f"{i}_{j}"] = bond_cls(i, j, self.r_max, self.K)

        self._einsum_str = self._build_einsum_string()
        self.reset_parameters()

    def reset_parameters(self):
        for tensor in self.local_tensors:
            nn.init.normal_(tensor, mean=0.0, std=0.1)
        for bond in self.bonds.values():
            bond.reset_parameters()

    def _build_einsum_string(self) -> str:
        m = self.m
        labels = iter(string.ascii_lowercase)
        phys_labels = [next(labels) for _ in range(m)]
        bond_labels: dict[tuple[int, int], tuple[str, str]] = {}
        for i in range(m):
            for j in range(i + 1, m):
                bond_labels[(i, j)] = (next(labels), next(labels))

        local_operands = []
        for i in range(m):
            indices = [phys_labels[i]]
            for j in range(m):
                if j == i:
                    continue
                edge = (min(i, j), max(i, j))
                li, lj = bond_labels[edge]
                indices.append(li if i < j else lj)
            local_operands.append("".join(indices))

        bond_operands = []
        for i in range(m):
            for j in range(i + 1, m):
                li, lj = bond_labels[(i, j)]
                bond_operands.append(li + lj)

        output = "".join(phys_labels)
        return ",".join(local_operands + bond_operands) + "->" + output

    def contract(self) -> torch.Tensor:
        operands = list(self.local_tensors)
        operands += [bond.matrix() for bond in self.bonds.values()]
        return torch.einsum(self._einsum_str, *operands)

    def forward(self) -> torch.Tensor:
        return self.contract()

    def reconstruction_loss(self, p: torch.Tensor) -> torch.Tensor:
        return torch.sum((p - self.contract()) ** 2)

    def penalty(self) -> torch.Tensor:
        return sum(bond.frobenius_penalty() for bond in self.bonds.values())

    def loss(self, p: torch.Tensor, beta: float) -> torch.Tensor:
        return self.reconstruction_loss(p) + beta * self.penalty()

    @torch.no_grad()
    def bond_nuclear_norms(self) -> dict[tuple[int, int], float]:
        return {
            tuple(int(x) for x in k.split("_")): b.nuclear_norm().item()
            for k, b in self.bonds.items()
        }

    @torch.no_grad()
    def bond_effective_ranks(self, tol: float = 1e-6) -> dict[tuple[int, int], int]:
        return {
            tuple(int(x) for x in k.split("_")): b.effective_rank(tol)
            for k, b in self.bonds.items()
        }

    @torch.no_grad()
    def effective_graph(
        self, rank_tol: float = 1e-6, min_rank: int = 1
    ) -> list[tuple[int, int]]:
        return [
            e for e, r in self.bond_effective_ranks(rank_tol).items() if r >= min_rank
        ]

    def extra_repr(self) -> str:
        return f"dims={self.dims}, r_max={self.r_max}, K={self.K}, m={self.m}"
