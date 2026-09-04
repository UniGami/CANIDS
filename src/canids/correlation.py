"""Offline partner correlation graph, built once from normal data only.

Per pair of signals, computes Pearson correlation over the grid-aligned
value channels and keeps an edge only if it clears the strength cutoff in
every fold of a fold-split of the data (fold-stability filtering). This
guards against edges that reflect one operating-state-specific coincidence
rather than a functional dependency that holds throughout normal operation
(see the correlation-graph-validity limitation discussed and agreed with the
user in PLAN.md).

Required input to replay attribution (attribution/rules.py, Step 9): two
signals moving together under normal operation are "partners," and a
residual spike shared by both partners at the same time (absent a
plateau/drift signature) is the replay signature.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from canids.config import CORRELATION_FOLDS, CORRELATION_STRENGTH_CUTOFF
from canids.registry import Registry


@dataclass(frozen=True)
class CorrelationEdge:
    signal_a: int  # registry signal_index
    signal_b: int
    strength: float  # correlation coefficient, averaged over folds where it passed the cutoff
    fold_agreement: int  # number of folds (out of n_folds) where |correlation| >= strength_cutoff


class CorrelationGraph:
    def __init__(self, edges: list[CorrelationEdge], n_folds: int):
        self.edges = edges
        self.n_folds = n_folds
        self._adjacency: dict[int, list[CorrelationEdge]] = {}
        for edge in edges:
            self._adjacency.setdefault(edge.signal_a, []).append(edge)
            self._adjacency.setdefault(edge.signal_b, []).append(edge)

    def partners(self, signal_index: int) -> list[CorrelationEdge]:
        return self._adjacency.get(signal_index, [])

    def partner_indices(self, signal_index: int) -> list[int]:
        return [
            edge.signal_b if edge.signal_a == signal_index else edge.signal_a
            for edge in self.partners(signal_index)
        ]

    def is_partner(self, signal_a: int, signal_b: int) -> bool:
        return signal_b in self.partner_indices(signal_a)

    def save(self, path: Path) -> None:
        payload = {"n_folds": self.n_folds, "edges": [asdict(e) for e in self.edges]}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CorrelationGraph":
        payload = json.loads(Path(path).read_text())
        edges = [CorrelationEdge(**row) for row in payload["edges"]]
        return cls(edges=edges, n_folds=payload["n_folds"])


def _value_matrix(joint_vector: np.ndarray, registry: Registry) -> np.ndarray:
    """Extract just the value channels (n_ticks, n_signals), dropping
    staleness — correlation is computed over signal values only.
    """
    value_indices = [entry.value_index for entry in registry.entries]
    return joint_vector[:, value_indices]


def build_correlation_graph(
    joint_vector: np.ndarray,
    registry: Registry,
    n_folds: int = CORRELATION_FOLDS,
    strength_cutoff: float = CORRELATION_STRENGTH_CUTOFF,
    min_fold_agreement: int | None = None,
) -> CorrelationGraph:
    """Build the fold-stability-filtered correlation graph from normal-only,
    grid-aligned data (see data/windowing.build_joint_vector).

    The tick series is split into n_folds contiguous folds; a Pearson
    correlation matrix is computed per fold, and an edge is kept only if it
    clears strength_cutoff in at least min_fold_agreement folds (default:
    every fold — the strictest, most defensible reading of "stable
    dependency," since this is exactly the mitigation for the risk that a
    correlation only holds under one operating state). Loosen
    min_fold_agreement for a sensitivity comparison, not as the default.
    """
    if min_fold_agreement is None:
        min_fold_agreement = n_folds

    values = _value_matrix(joint_vector, registry)
    # Drop the pre-first-transmission warm-up ticks (see data/grid.py):
    # np.corrcoef poisons its entire output with NaN if any input row has
    # one, so this data artifact would otherwise silently zero out real
    # correlations for whichever fold it lands in.
    values = values[~np.isnan(values).any(axis=1)]
    n_ticks, n_signals = values.shape
    fold_bounds = np.linspace(0, n_ticks, n_folds + 1, dtype=int)

    n_folds_passed = np.zeros((n_signals, n_signals), dtype=int)
    strength_sum = np.zeros((n_signals, n_signals), dtype=float)

    for i in range(n_folds):
        fold_values = values[fold_bounds[i] : fold_bounds[i + 1]]
        corr = np.corrcoef(fold_values, rowvar=False)
        # A constant signal within a fold (zero variance) makes its
        # correlation with everything undefined; treat that as no evidence
        # of correlation for this fold rather than propagating NaN.
        corr = np.nan_to_num(corr, nan=0.0)
        passed = np.abs(corr) >= strength_cutoff
        n_folds_passed += passed
        strength_sum += np.where(passed, corr, 0.0)

    edges = []
    for a in range(n_signals):
        for b in range(a + 1, n_signals):
            if n_folds_passed[a, b] >= min_fold_agreement:
                avg_strength = strength_sum[a, b] / max(n_folds_passed[a, b], 1)
                edges.append(
                    CorrelationEdge(
                        signal_a=a,
                        signal_b=b,
                        strength=float(avg_strength),
                        fold_agreement=int(n_folds_passed[a, b]),
                    )
                )

    return CorrelationGraph(edges=edges, n_folds=n_folds)
