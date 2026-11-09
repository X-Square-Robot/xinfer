import pytest

from xinfer.models.interface import Stage, StageGraph


def test_order_follows_dependencies_not_declaration():
    graph = StageGraph(
        Stage("action", "denoise", depends_on="prefill"),
        Stage("vision", "encode"),
        Stage("prefill", "prefill", depends_on="vision"),
    )
    assert [s.name for s in graph.order] == ["vision", "prefill", "action"]
    assert graph.consumers("vision") == ("prefill",)
    assert graph["prefill"].depends_on == ("vision",)


@pytest.mark.parametrize("stages", [
    (Stage("a", "f"), Stage("b", "g")),
    (Stage("a", "f"), Stage("b", "g", depends_on="a"), Stage("c", "h", depends_on="a")),
], ids=["disconnected", "fork"])
def test_rejects_multiple_terminal_stages(stages):
    with pytest.raises(ValueError, match="exactly one terminal stage"):
        StageGraph(*stages)


def test_accepts_branches_with_one_terminal_stage():
    graph = StageGraph(
        Stage("a", "f"),
        Stage("b", "g", depends_on="a"),
        Stage("c", "h", depends_on="a"),
        Stage("d", "j", depends_on=("b", "c")),
    )
    assert [s.name for s in graph.order] == ["a", "b", "c", "d"]
    assert graph.consumers("a") == ("b", "c")
    assert graph.consumers("b") == graph.consumers("c") == ("d",)


def test_rejects_unknown_dependency():
    with pytest.raises(ValueError, match="unknown stages"):
        StageGraph(Stage("prefill", "prefill", depends_on="vision"))


def test_rejects_cycle():
    with pytest.raises(ValueError, match="cycle"):
        StageGraph(Stage("a", "f", depends_on="b"), Stage("b", "g", depends_on="a"))


def test_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        StageGraph(Stage("a", "f"), Stage("a", "g"))
