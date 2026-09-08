"""Graph recovery and verification utilities.

Contains nuclear-norm thresholding for effective graph extraction and
conditional mutual information for post-hoc verification.
"""

from __future__ import annotations

from typing import Optional
import torch
import numpy as np


# -------------------------------------------------------------------- #
#  Nuclear norm thresholding                                           #
# -------------------------------------------------------------------- #

# def select_tau(
#     nuclear_norms: dict[tuple[int, int], float],
#     min_gap_ratio: float = 2.0,
# ) -> Optional[float]:
#     """Select threshold via maximum gap ratio in sorted nuclear norms.

#     Sorts norms descending. Finds the consecutive pair with the largest
#     ratio exceeding min_gap_ratio. Returns the geometric mean of that
#     pair as tau. Returns None if no gap exceeds min_gap_ratio (keep all).

#     The geometric mean sits at the center of the gap in log-space,
#     maximizing the margin to both clusters.
#     """
#     sorted_norms = sorted(nuclear_norms.values(), reverse=True)
#     if len(sorted_norms) <= 1:
#         return None

#     best_ratio = min_gap_ratio
#     best_gap = None

#     for k in range(len(sorted_norms) - 1):
#         upper = sorted_norms[k]
#         lower = sorted_norms[k + 1]
#         if lower > 1e-12:
#             ratio = upper / lower
#             if ratio > best_ratio:
#                 best_ratio = ratio
#                 best_gap = (upper, lower)

#     if best_gap is None:
#         return None

#     return (best_gap[0] * best_gap[1]) ** 0.5

def select_tau(
    nuclear_norms: dict[tuple[int, int], float],
    min_variance_ratio: float = 0.5,
) -> float | None:
    """Select threshold via Otsu's method in log-space.

    Splits at the point maximizing between-class variance of log-norms.
    Returns the geometric mean of the boundary pair, or None if the
    data is not sufficiently bimodal.
    """
    values = sorted(nuclear_norms.values())
    n = len(values)
    if n <= 1:
        return None

    # Floor zeros at eps to avoid -inf in log.
    # This also ensures Otsu captures the zero-vs-nonzero gap naturally.
    eps = 1e-12
    clipped = np.maximum(np.array(values, dtype=float), eps)
    log_vals = np.log(clipped)

    total_var = log_vals.var()
    if total_var < 1e-24:
        return None  # All values are identical (e.g., all zeros)

    best_eta = -1.0
    best_split = 1

    for k in range(1, n):
        w1, w2 = k / n, (n - k) / n
        mu1, mu2 = log_vals[:k].mean(), log_vals[k:].mean()
        eta = w1 * w2 * (mu1 - mu2) ** 2 / total_var
        if eta > best_eta:
            best_eta = eta
            best_split = k

    if best_eta < min_variance_ratio:
        return None

    lower = clipped[best_split - 1]
    upper = clipped[best_split]
    return (lower * upper) ** 0.5

def recover_graph(
    nuclear_norms: dict[tuple[int, int], float],
    tau: Optional[float],
) -> list[tuple[int, int]]:
    """Recover effective graph by thresholding nuclear norms."""
    if tau is None:
        return list(nuclear_norms.keys())
    return [e for e, norm in nuclear_norms.items() if norm > tau]


# -------------------------------------------------------------------- #
#  Conditional mutual information                                      #
# -------------------------------------------------------------------- #

def conditional_mi(p: torch.Tensor, i: int, j: int) -> float:
    """I_C(V_i : V_j | V_rest) for a joint distribution tensor.

    Uses: I(i:j|rest) = H(i,rest) + H(j,rest) - H(rest) - H(all).
    Assumes p is a normalized probability tensor.
    """
    eps = 1e-12

    # H(i, rest): marginalize out j
    p_i_rest = p.sum(dim=j)
    H_i_rest = -(p_i_rest * torch.log(p_i_rest + eps)).sum()

    # H(j, rest): marginalize out i
    p_j_rest = p.sum(dim=i)
    H_j_rest = -(p_j_rest * torch.log(p_j_rest + eps)).sum()

    # H(rest): marginalize out both i and j (descending order to avoid shift)
    dims_to_sum = sorted([i, j], reverse=True)
    p_rest = p
    for d in dims_to_sum:
        p_rest = p_rest.sum(dim=d)
    H_rest = -(p_rest * torch.log(p_rest + eps)).sum()

    # H(all)
    H_all = -(p * torch.log(p + eps)).sum()

    return (H_i_rest + H_j_rest - H_rest - H_all).item()


def all_pairwise_cmi(p: torch.Tensor) -> dict[tuple[int, int], float]:
    """Compute I_C for all C(m,2) pairs."""
    m = p.dim()
    result = {}
    for i in range(m):
        for j in range(i + 1, m):
            result[(i, j)] = conditional_mi(p, i, j)
    return result

def recover_graph_by_cmi(
    p: torch.Tensor,
    tol: float = 1e-3,
) -> list[tuple[int, int]]:
    """Recover moral graph via conditional mutual information thresholding.

    An edge (i,j) is in the moral graph iff I_C(V_i : V_j | V_rest) > tol.
    This operates directly on the distribution and is invariant to the
    TN's internal factorization choices (unlike nuclear norm thresholding,
    which can fail when the optimizer finds an isomorphic local minimum).

    For approximate reconstructions, tol should be set above the
    expected I_C noise floor. The Fannes-Audenaert continuity bound
    f(epsilon) provides a principled choice; for now, a fixed threshold
    is used.
    """
    cmi = all_pairwise_cmi(p)
    return [e for e, val in cmi.items() if val > tol]

def recover_graph_by_rank(
    svals: dict[tuple[int, int], torch.Tensor],
    rel_tol: float = 0.01,
    abs_floor: float = 1e-10,
) -> set[tuple[int, int]]:
    """Recover moral graph via effective rank > 1.

    A singular value counts as 'significant' if:
      s_i > rel_tol * s_max   AND   s_i > abs_floor

    rel_tol: fraction of largest singular value (default 1%).
    abs_floor: pure numerical noise floor.
    """
    edges = set()
    for edge, s in svals.items():
        s = s.tolist()
        s_max = max(s) if s else 0.0
        if s_max < abs_floor:
            continue  # entirely degenerate bond
        n_significant = sum(
            1 for si in s if si > rel_tol * s_max and si > abs_floor
        )
        if n_significant > 1:
            edges.add(edge)
    return edges