"""Unit tests for predictive-validity runtime device resolution."""

from __future__ import annotations

from types import SimpleNamespace

from experiments.run_predictive_validity import _resolve_runtime_device


class _BackendFlag:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _TorchStub:
    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = _BackendFlag(cuda_available)
        self.backends = SimpleNamespace(mps=_BackendFlag(mps_available))


def test_auto_prefers_mps_when_available() -> None:
    torch_stub = _TorchStub(cuda_available=True, mps_available=True)
    assert _resolve_runtime_device("auto", torch_stub) == "mps"


def test_auto_uses_cuda_when_no_mps() -> None:
    torch_stub = _TorchStub(cuda_available=True, mps_available=False)
    assert _resolve_runtime_device("auto", torch_stub) == "cuda"


def test_requested_mps_falls_back_to_cuda() -> None:
    torch_stub = _TorchStub(cuda_available=True, mps_available=False)
    assert _resolve_runtime_device("mps", torch_stub) == "cuda"


def test_requested_cuda_falls_back_to_cpu() -> None:
    torch_stub = _TorchStub(cuda_available=False, mps_available=False)
    assert _resolve_runtime_device("cuda", torch_stub) == "cpu"


def test_unknown_device_falls_back_to_auto() -> None:
    torch_stub = _TorchStub(cuda_available=False, mps_available=True)
    assert _resolve_runtime_device("weird-device", torch_stub) == "mps"
