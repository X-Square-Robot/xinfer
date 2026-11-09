"""benchmarks/performance.py: the capacity rule and the table row (CPU, no service)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import performance  # noqa: E402


def _round(p50, p95=None, errors=0, missing=0, throughput=10.0):
    return {"errors": errors, "missing": missing, "throughput_rps": throughput,
            "server_infer_ms": {"p50": p50, "p95": p95 or p50 + 10}}


def test_a_count_passes_when_the_median_round_is_below_the_limit_and_no_round_has_errors():
    assert performance.passes([_round(100), _round(149.9), _round(170)])  # median 149.9
    assert not performance.passes([_round(100), _round(150.0), _round(170)])
    assert performance.passes([_round(100), _round(140), _round(150), _round(170)])  # median 145
    assert not performance.passes([_round(100), _round(100, errors=1), _round(100)])
    assert not performance.passes([_round(100), _round(100, missing=1), _round(100)])
    assert not performance.passes([])


def test_round_statistics_give_the_median_and_the_worst_round():
    stats = performance.round_statistics([_round(90, 112), _round(140, 150), _round(105, 114)])
    assert stats == {"p50_median": 105.0, "p95_median": 114.0, "p50_worst": 140, "p95_worst": 150}


def test_the_capacity_is_the_largest_count_with_every_smaller_count_passing():
    ok, bad = [_round(100)] * 3, [_round(160)] * 3
    assert performance.capacity({3: ok, 4: ok, 5: bad}) == (4, True)
    assert performance.capacity({3: ok, 4: bad, 5: ok}) == (3, True)
    assert performance.capacity({3: ok, 4: ok}) == (4, False)  # no failing count measured: a lower bound
    assert performance.capacity({3: bad}) == (None, True)


def test_the_row_reports_the_capacity_point_and_the_saturation_throughput():
    result = {"name": "M · BF16 · GPU", "latency": {"total_ms": {"p50": 44.21}},
              "load": {"3": [_round(97, 103, throughput=16.7)] * 3,
                       "4": [_round(90, 112, throughput=22.2), _round(140, 150, throughput=22.2),
                             _round(105, 114, throughput=22.2)],
                       "5": [_round(160, throughput=26.4)] * 3},
              "capacity": (4, True)}
    assert performance.row(result) == "| M · BF16 · GPU | 44.2 ms | 4 | 105 ms | 114 ms | 26.4 req/s |"
    result["capacity"] = (5, False)
    result["load"]["5"] = [_round(120, throughput=26.4)] * 3
    assert "| ≥ 5 |" in performance.row(result)
