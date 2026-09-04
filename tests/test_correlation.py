import pytest

from canids.correlation import build_correlation_graph
from canids.data.grid import align_to_grid
from canids.data.synthetic import generate_normal, write_csv
from canids.data.windowing import build_joint_vector
from canids.registry import build_registry


@pytest.fixture(scope="module")
def normal_graph_fixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("correlation")
    df = generate_normal(duration_seconds=60.0, seed=1)
    path = tmp_path / "normal.csv"
    write_csv(df, path)

    registry = build_registry([path])
    alignment = align_to_grid(df, registry, step=0.01)
    joint = build_joint_vector(alignment, registry)
    graph = build_correlation_graph(joint, registry)
    return registry, graph


def _idx(registry, can_id, slot):
    return registry.entry(can_id, slot).signal_index


def test_known_correlated_pair_is_a_partner(normal_graph_fixture):
    registry, graph = normal_graph_fixture
    a = _idx(registry, "ID_A", 1)
    b = _idx(registry, "ID_B", 1)

    assert graph.is_partner(a, b)
    edge = graph.partners(a)[0]
    assert abs(edge.strength) > 0.7


def test_known_correlated_group_is_mutually_connected(normal_graph_fixture):
    registry, graph = normal_graph_fixture
    c2 = _idx(registry, "ID_C", 2)
    c3 = _idx(registry, "ID_C", 3)
    d1 = _idx(registry, "ID_D", 1)

    assert graph.is_partner(c2, c3)
    assert graph.is_partner(c2, d1)
    assert graph.is_partner(c3, d1)


def test_uncorrelated_signals_have_no_partners(normal_graph_fixture):
    registry, graph = normal_graph_fixture
    a2 = _idx(registry, "ID_A", 2)  # phase_mid, unique latent
    c1 = _idx(registry, "ID_C", 1)  # phase_slow, unique latent

    assert graph.partner_indices(a2) == []
    assert graph.partner_indices(c1) == []
    assert not graph.is_partner(a2, c1)


def test_fold_agreement_is_full_for_stable_synthetic_correlations(normal_graph_fixture):
    registry, graph = normal_graph_fixture
    a = _idx(registry, "ID_A", 1)
    edge = graph.partners(a)[0]
    assert edge.fold_agreement == graph.n_folds


def test_looser_fold_agreement_never_loses_edges(normal_graph_fixture):
    registry, graph = normal_graph_fixture
    df = generate_normal(duration_seconds=60.0, seed=1)
    alignment = align_to_grid(df, registry, step=0.01)
    joint = build_joint_vector(alignment, registry)

    loose_graph = build_correlation_graph(joint, registry, min_fold_agreement=1)
    strict_pairs = {(e.signal_a, e.signal_b) for e in graph.edges}
    loose_pairs = {(e.signal_a, e.signal_b) for e in loose_graph.edges}
    assert strict_pairs <= loose_pairs


def test_correlation_graph_save_and_load_roundtrip(tmp_path, normal_graph_fixture):
    _, graph = normal_graph_fixture
    path = tmp_path / "correlation.json"
    graph.save(path)
    loaded = graph.__class__.load(path)

    assert loaded.n_folds == graph.n_folds
    assert {(e.signal_a, e.signal_b) for e in loaded.edges} == {
        (e.signal_a, e.signal_b) for e in graph.edges
    }
