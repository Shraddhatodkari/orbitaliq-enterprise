import numpy as np
import pytest
import torch

from orbitaliq.nvidia.vision_model import (
    FacilityChangeCNN,
    VisionInferenceEngine,
    compute_change_indices,
    export_to_onnx,
    resolve_device,
)


def test_resolve_device_cpu_explicit():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_never_raises():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda"}


def test_resolve_device_cuda_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == torch.device("cpu")


def test_compute_change_indices_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compute_change_indices(np.zeros((4, 64, 64), dtype=np.float32), np.zeros((4, 64, 64), dtype=np.float32))
    with pytest.raises(ValueError):
        compute_change_indices(np.zeros((3, 64, 64), dtype=np.float32), np.zeros((3, 32, 32), dtype=np.float32))


def test_compute_change_indices_brighter_current_has_high_construction_index():
    prior = np.full((3, 8, 8), 0.3, dtype=np.float32)
    current = np.full((3, 8, 8), 0.7, dtype=np.float32)
    indices = compute_change_indices(current, prior)
    assert indices["construction_index"] > 0.9


def test_compute_change_indices_greener_prior_has_high_clearing_index():
    prior = np.full((3, 8, 8), 0.3, dtype=np.float32)
    prior[1, :, :] = 0.9  # prior was lush green
    current = np.full((3, 8, 8), 0.3, dtype=np.float32)  # current has lost the green
    indices = compute_change_indices(current, prior)
    assert indices["clearing_index"] > 0.9


def test_compute_change_indices_identical_tiles_have_zero_change_energy():
    tile = np.full((3, 8, 8), 0.5, dtype=np.float32)
    indices = compute_change_indices(tile, tile.copy())
    assert indices["change_energy"] == pytest.approx(0.0, abs=1e-6)


def test_vision_inference_engine_output_ranges_and_metadata():
    engine = VisionInferenceEngine(device_preference="cpu", seed=1)
    current = np.random.default_rng(0).uniform(0, 1, size=(3, 64, 64)).astype(np.float32)
    prior = np.random.default_rng(1).uniform(0, 1, size=(3, 64, 64)).astype(np.float32)
    findings = engine.analyze_change_pair(current, prior)

    for value in [
        findings.construction_expansion_signal,
        findings.vegetation_clearing_signal,
        findings.overall_change_score,
        findings.change_energy,
    ]:
        assert 0.0 <= value <= 1.0

    assert findings.device_used == "cpu"
    assert findings.model_finetuned is False

    d = findings.as_dict()
    assert set(d.keys()) == {
        "construction_expansion_signal",
        "vegetation_clearing_signal",
        "overall_change_score",
        "change_energy",
        "device_used",
        "model_finetuned",
        "status",
    }


def test_vision_inference_engine_is_deterministic_for_same_pair():
    engine = VisionInferenceEngine(device_preference="cpu", seed=7)
    current = np.random.default_rng(0).uniform(0, 1, size=(3, 64, 64)).astype(np.float32)
    prior = np.random.default_rng(1).uniform(0, 1, size=(3, 64, 64)).astype(np.float32)
    f1 = engine.analyze_change_pair(current, prior)
    f2 = engine.analyze_change_pair(current, prior)
    assert f1.as_dict() == f2.as_dict()


def test_vision_inference_engine_flags_higher_construction_signal_for_bright_new_patch():
    engine = VisionInferenceEngine(device_preference="cpu", seed=3, cnn_weight=0.0)
    prior = np.full((3, 64, 64), 0.3, dtype=np.float32)
    unchanged_current = prior.copy()
    expanded_current = prior.copy()
    expanded_current[:, 20:44, 20:44] = 0.9  # bright new bare-earth/concrete patch

    baseline_findings = engine.analyze_change_pair(unchanged_current, prior)
    expanded_findings = engine.analyze_change_pair(expanded_current, prior)
    assert expanded_findings.construction_expansion_signal > baseline_findings.construction_expansion_signal


def test_analyze_batch_returns_one_result_per_pair():
    engine = VisionInferenceEngine(device_preference="cpu")
    pairs = [
        (
            np.random.default_rng(i).uniform(0, 1, size=(3, 64, 64)).astype(np.float32),
            np.random.default_rng(i + 10).uniform(0, 1, size=(3, 64, 64)).astype(np.float32),
        )
        for i in range(3)
    ]
    results = engine.analyze_batch(pairs)
    assert len(results) == 3


def test_export_to_onnx_never_raises(tmp_path):
    model = FacilityChangeCNN()
    output_path = str(tmp_path / "model.onnx")
    result = export_to_onnx(model, output_path)
    assert "success" in result
    assert isinstance(result["success"], bool)
    if result["success"]:
        import os

        assert os.path.isfile(output_path)
