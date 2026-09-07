"""Tiny-file fault tests for atomic writer initialization and publication."""

import builtins

import pytest
import torch
from safetensors.torch import load_file, save_file

from lib import incremental_writer, streaming_save


@pytest.mark.parametrize("adapter", ["incremental", "materialization"])
@pytest.mark.parametrize("failure", ["open", "header_length", "header", "seek", "preallocate"])
# AC: @saved-model-artifact-safety ac-no-partial-publication
# AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
def test_initial_io_failure_closes_and_removes_temp(tmp_path, monkeypatch, adapter, failure):
    target = tmp_path / "model.safetensors"
    save_file({"old": torch.ones(2)}, str(target))
    original = target.read_bytes()
    opened = []

    class FailingFile:
        def __init__(self, file):
            self.file = file
            self.writes = 0

        def write(self, data):
            self.writes += 1
            if self.writes == {"header_length": 1, "header": 2, "preallocate": 3}.get(failure):
                raise OSError("injected disk failure")
            return self.file.write(data)

        def seek(self, offset):
            if failure == "seek":
                raise OSError("injected disk failure")
            return self.file.seek(offset)

        def close(self):
            return self.file.close()

    def failing_open(*args, **kwargs):
        if failure == "open":
            raise OSError("injected disk failure")
        file = builtins.open(*args, **kwargs)
        opened.append(file)
        return FailingFile(file)

    module = incremental_writer if adapter == "incremental" else streaming_save
    monkeypatch.setattr(module, "open", failing_open, raising=False)
    manifest = {"new": (torch.float32, (2,))}
    with pytest.raises(OSError, match="injected disk failure"):
        if adapter == "incremental":
            incremental_writer.IncrementalWriter(manifest, str(target))
        else:
            streaming_save.MaterializationSink().open(manifest, str(target))
    try:
        assert all(file.closed for file in opened), "failed initialization leaked an open file"
        assert not list(tmp_path.glob(".ecaj_tmp_*")), "failed initialization leaked temp artifact"
        assert target.read_bytes() == original
        assert torch.equal(load_file(str(target))["old"], torch.ones(2))
    finally:
        for file in opened:
            file.close()
