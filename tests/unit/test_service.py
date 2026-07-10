from pathlib import Path
import os

from ipasc_iqa.gui import service


def test_ensure_writable_model_cache_sets_missing_cache_envs(tmp_path: Path, monkeypatch) -> None:
    ws = service.Workspace.resolve(tmp_path / "ws", tmp_path / "data")
    ws.ensure()

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    service.ensure_writable_model_cache(ws)

    assert os.environ["XDG_CACHE_HOME"] == str(ws.model_cache_dir / "xdg")
    assert os.environ["TORCH_HOME"] == str(ws.model_cache_dir / "torch")
    assert os.environ["HF_HOME"] == str(ws.model_cache_dir / "hf")


def test_ensure_writable_model_cache_overrides_unwritable_cache(tmp_path: Path, monkeypatch) -> None:
    ws = service.Workspace.resolve(tmp_path / "ws", tmp_path / "data")
    ws.ensure()

    monkeypatch.setenv("XDG_CACHE_HOME", "/opt/.cache")
    monkeypatch.setenv("TORCH_HOME", "/opt/.cache/torch")
    monkeypatch.setenv("HF_HOME", "/opt/.cache/hf")
    monkeypatch.setattr(service, "_can_write_dir", lambda path: False)

    service.ensure_writable_model_cache(ws)

    assert os.environ["XDG_CACHE_HOME"] == str(ws.model_cache_dir / "xdg")
    assert os.environ["TORCH_HOME"] == str(ws.model_cache_dir / "torch")
    assert os.environ["HF_HOME"] == str(ws.model_cache_dir / "hf")


def test_ensure_writable_model_cache_keeps_existing_writable_cache(tmp_path: Path, monkeypatch) -> None:
    ws = service.Workspace.resolve(tmp_path / "ws", tmp_path / "data")
    ws.ensure()

    existing = tmp_path / "existing_cache"
    (existing / "xdg").mkdir(parents=True)
    (existing / "torch").mkdir(parents=True)
    (existing / "hf").mkdir(parents=True)

    monkeypatch.setenv("XDG_CACHE_HOME", str(existing / "xdg"))
    monkeypatch.setenv("TORCH_HOME", str(existing / "torch"))
    monkeypatch.setenv("HF_HOME", str(existing / "hf"))

    service.ensure_writable_model_cache(ws)

    assert os.environ["XDG_CACHE_HOME"] == str(existing / "xdg")
    assert os.environ["TORCH_HOME"] == str(existing / "torch")
    assert os.environ["HF_HOME"] == str(existing / "hf")