"""Uvicorn entrypoint for the GUI (``ipasc_iqa-gui`` console script)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the ipasc_iqa web GUI.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("IPASC_IQA_WORKSPACE", ".")),
        help="Directory where run results are written (results/ folder; default: cwd).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            Path(os.environ["IPASC_IQA_DATA_DIR"])
            if os.environ.get("IPASC_IQA_DATA_DIR")
            else None
        ),
        help="Managed directory for downloaded datasets (default: platform data dir).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args()

    # Pass config to the app factory via env (works with uvicorn's reload subprocess).
    os.environ["IPASC_IQA_WORKSPACE"] = str(args.workspace.resolve())
    if args.data_dir is not None:
        os.environ["IPASC_IQA_DATA_DIR"] = str(args.data_dir.resolve())

    import uvicorn

    uvicorn.run(
        "ipasc_iqa.gui.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
