import pytest
import torch
from safetensors.torch import save_file

from xinfer.loader import SafetensorsCheckpoint, Source, load_mapped


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        with torch.device("meta"):
            self.proj = torch.nn.Linear(4, 6, bias=False)
            self.norm = torch.nn.Parameter(torch.empty(4))
        self.register_buffer("derived", torch.arange(3.0), persistent=False)


@pytest.fixture
def checkpoint(tmp_path):
    tensors = {
        "q.weight": torch.full((2, 4), 1.0),
        "k.weight": torch.full((2, 4), 2.0),
        "v.weight": torch.full((2, 4), 3.0),
        "norm": torch.ones(4, dtype=torch.float32),
        "unused": torch.zeros(1),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return SafetensorsCheckpoint(tmp_path)


def _mapping():
    return {
        "proj.weight": Source(("q.weight", "k.weight", "v.weight"), torch.bfloat16,
                              combine=lambda ts: torch.cat(ts, 0)),
        "norm": Source.one("norm", torch.bfloat16),
    }


def test_strict_mapped_load(checkpoint):
    model = Tiny()
    report = load_mapped(model, _mapping(), checkpoint, device="cpu")
    assert report.unused_keys == ("unused",)
    assert model.proj.weight.dtype == torch.bfloat16
    assert model.proj.weight[:, 0].tolist() == [1, 1, 2, 2, 3, 3]
    assert not model.proj.weight.requires_grad


def test_unmapped_target_is_an_error(checkpoint):
    mapping = _mapping()
    del mapping["norm"]
    with pytest.raises(KeyError, match="no checkpoint source"):
        load_mapped(Tiny(), mapping, checkpoint, device="cpu")


def test_shape_mismatch_and_missing_key(checkpoint):
    mapping = _mapping()
    mapping["norm"] = Source.one("q.weight", torch.float32)
    with pytest.raises(ValueError, match="shape"):
        load_mapped(Tiny(), mapping, checkpoint, device="cpu")
    mapping["norm"] = Source.one("absent", torch.float32)
    with pytest.raises(KeyError, match="lacks"):
        load_mapped(Tiny(), mapping, checkpoint, device="cpu")
