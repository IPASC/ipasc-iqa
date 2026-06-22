"""Environment checks for ipasc_iqa installations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module, metadata, util
from typing import Literal

CheckStatus = Literal["ok", "warning", "error", "skipped"]


@dataclass(frozen=True)
class CheckResult:
    """Result of a single dependency/environment check."""

    name: str
    group: str
    status: CheckStatus
    message: str
    version: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


_CORE_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "numpy"),
    ("scipy", "scipy", "scipy"),
    ("scikit-image", "skimage", "scikit-image"),
    ("pydantic", "pydantic", "pydantic"),
    ("typer", "typer", "typer"),
    ("jsonschema", "jsonschema", "jsonschema"),
    ("diskcache", "diskcache", "diskcache"),
    ("platformdirs", "platformdirs", "platformdirs"),
    ("tqdm", "tqdm", "tqdm"),
    ("pillow", "PIL", "pillow"),
)

_GUI_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("fastapi", "fastapi", "fastapi"),
    ("uvicorn", "uvicorn", "uvicorn"),
    ("jinja2", "jinja2", "jinja2"),
    ("python-multipart", "multipart", "python-multipart"),
    ("bcrypt", "bcrypt", "bcrypt"),
)

_DEEP_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("torch", "torch", "torch"),
    ("torchvision", "torchvision", "torchvision"),
    ("lpips", "lpips", "lpips"),
    ("DISTS-pytorch", "DISTS_pytorch", "DISTS-pytorch"),
    ("piq", "piq", "piq"),
)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _check_import(
    *,
    name: str,
    module: str,
    distribution: str,
    group: str,
    required: bool,
    install_hint: str | None = None,
    import_check: bool = True,
) -> CheckResult:
    version = _version(distribution)
    if util.find_spec(module) is None:
        status: CheckStatus = "error" if required else "skipped"
        suggestion = install_hint or f"Install missing dependency '{distribution}'."
        return CheckResult(
            name=name,
            group=group,
            status=status,
            version=version,
            message=f"Missing import '{module}'.",
            suggestion=suggestion,
        )

    if import_check:
        try:
            import_module(module)
        except Exception as exc:
            status = "error" if required else "warning"
            return CheckResult(
                name=name,
                group=group,
                status=status,
                version=version,
                message=f"Import failed: {exc}",
                suggestion=install_hint,
            )

    message = "Import OK." if import_check else "Installed."
    return CheckResult(
        name=name,
        group=group,
        status="ok",
        version=version,
        message=message,
    )


def _check_torch_numpy_interop(required: bool) -> CheckResult:
    numpy_version = _version("numpy")
    torch_version = _version("torch")
    if util.find_spec("torch") is None:
        status: CheckStatus = "error" if required else "skipped"
        return CheckResult(
            name="torch-numpy interop",
            group="deep",
            status=status,
            version=f"numpy={numpy_version or 'missing'}, torch=missing",
            message="Torch is not installed, so torch.from_numpy cannot be checked.",
            suggestion="Install deep dependencies with: pip install 'ipasc_iqa[deep]'",
        )

    try:
        import numpy as np
        import torch

        x = np.zeros((2, 2), dtype=np.float32)
        torch.from_numpy(x)
    except Exception as exc:
        return CheckResult(
            name="torch-numpy interop",
            group="deep",
            status="error",
            version=f"numpy={numpy_version or 'unknown'}, torch={torch_version or 'unknown'}",
            message=f"torch.from_numpy failed: {exc}",
            suggestion=(
                "Use a compatible NumPy/PyTorch pair. For the current tested setup, "
                "try: pip install 'numpy>=1.26,<2'"
            ),
        )

    return CheckResult(
        name="torch-numpy interop",
        group="deep",
        status="ok",
        version=f"numpy={numpy_version or 'unknown'}, torch={torch_version or 'unknown'}",
        message="torch.from_numpy OK.",
    )


def run_doctor(*, require_gui: bool = False, require_deep: bool = False) -> list[CheckResult]:
    """Run dependency checks for the current Python environment.

    Core dependencies are always required. GUI/deep dependencies are probed by
    default, but missing optional dependencies are reported as ``skipped`` unless
    the corresponding ``require_*`` flag is true.
    """

    results: list[CheckResult] = []
    for name, module, distribution in _CORE_DEPENDENCIES:
        results.append(
            _check_import(
                name=name,
                module=module,
                distribution=distribution,
                group="core",
                required=True,
            )
        )

    for name, module, distribution in _GUI_DEPENDENCIES:
        results.append(
            _check_import(
                name=name,
                module=module,
                distribution=distribution,
                group="gui",
                required=require_gui,
                install_hint="Install GUI dependencies with: pip install 'ipasc_iqa[gui]'",
            )
        )

    for name, module, distribution in _DEEP_DEPENDENCIES:
        results.append(
            _check_import(
                name=name,
                module=module,
                distribution=distribution,
                group="deep",
                required=require_deep,
                install_hint="Install deep dependencies with: pip install 'ipasc_iqa[deep]'",
                import_check=False,
            )
        )

    results.append(_check_torch_numpy_interop(required=require_deep))
    return results


def has_errors(results: list[CheckResult]) -> bool:
    """Return True when any check failed with status ``error``."""

    return any(result.status == "error" for result in results)
