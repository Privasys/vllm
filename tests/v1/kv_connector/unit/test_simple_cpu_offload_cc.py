# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleCPUOffload under NVIDIA Confidential Computing.

The GPU refuses to register guest memory under CC (cudaHostRegister fails),
so the CPU offload buffers must stay unpinned there.
"""

import pytest
import torch

import vllm.v1.simple_kv_offload.worker as worker_mod
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


def _worker() -> SimpleCPUOffloadWorker:
    w = object.__new__(SimpleCPUOffloadWorker)
    w.num_cpu_blocks = 4
    w.cpu_hugepage_size = None
    w.load_stream = None
    w.store_stream = None
    w._pinned_host_ranges = []
    return w


def _refuse_registration(tensor: torch.Tensor) -> None:
    raise RuntimeError("cudaHostRegister failed: cudaError.???")


@pytest.mark.parametrize("confidential", [True, False])
def test_cpu_buffers_are_not_pinned_under_confidential_computing(
    monkeypatch: pytest.MonkeyPatch, confidential: bool
) -> None:
    pinned: list[torch.Tensor] = []

    def pin(tensor: torch.Tensor) -> None:
        if confidential:
            _refuse_registration(tensor)
        pinned.append(tensor)

    monkeypatch.setattr(worker_mod, "PIN_MEMORY", True)
    monkeypatch.setattr(worker_mod, "pin_tensor", pin)
    monkeypatch.setattr(
        current_platform,
        "is_confidential_compute",
        lambda: confidential,
        raising=False,
    )
    monkeypatch.setattr(worker_mod.DmaCopyBackend, "init", lambda *a, **k: None)

    w = _worker()
    caches = {"layer.0": torch.zeros(8, 2, 16, dtype=torch.float16)}
    w._init_cpu_mode(caches, 2 * 16 * 2, torch.device("cpu"))

    assert w.cpu_kv_caches["layer.0"].shape == (4, 2, 16)
    if confidential:
        assert pinned == [] and w._pinned_host_ranges == []
    else:
        assert pinned
