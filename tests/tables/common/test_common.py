from img2table.tables.common import (
    _cluster_values,
)


def test_cluster_values_uses_distinct_values_for_gap_computation() -> None:
    values = [0, 0, 0, 5, 20]

    assert _cluster_values(values=values, median_gap_multiple=1.5) == [0, 0, 0, 0, 1]
