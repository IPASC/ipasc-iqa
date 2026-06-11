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
        help="Directory containing datasets and a results/ folder (default: cwd).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args()

    # Pass workspace to the app factory via env (works with uvicorn's reload subprocess).
    os.environ["IPASC_IQA_WORKSPACE"] = str(args.workspace.resolve())

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
