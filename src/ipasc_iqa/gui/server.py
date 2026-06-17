"""Uvicorn entrypoint for the GUI (``ipasc_iqa-gui`` console script).

Usage:
    ipasc_iqa-gui [--workspace … --data-dir … --host … --port … --reload]
    ipasc_iqa-gui create-admin --email … --name … --password … [--workspace …]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _workspace_args(parser: argparse.ArgumentParser) -> None:
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
            Path(os.environ["IPASC_IQA_DATA_DIR"]) if os.environ.get("IPASC_IQA_DATA_DIR") else None
        ),
        help="Managed directory for downloaded datasets (default: platform data dir).",
    )


def _create_admin(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="ipasc_iqa-gui create-admin",
        description="Create or promote an administrator account.",
    )
    parser.add_argument("--email", required=True, help="Admin email (used to sign in).")
    parser.add_argument("--name", help="Display name (defaults to the email).")
    parser.add_argument(
        "--password",
        help="Password to set. Required for a new account; optional when promoting.",
    )
    _workspace_args(parser)
    args = parser.parse_args(argv)

    from ipasc_iqa.gui import auth
    from ipasc_iqa.gui.service import Workspace

    ws = Workspace.resolve(args.workspace, args.data_dir)
    ws.ensure()
    try:
        user = auth.promote_admin(ws, args.email, name=args.name, password=args.password)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(
        f"Admin ready: {user['name']} <{user['email']}> (uid {user['uid']}, role {user['role']})."
    )


def _serve(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Launch the ipasc_iqa web GUI.")
    _workspace_args(parser)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args(argv)

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


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "create-admin":
        _create_admin(argv[1:])
    else:
        _serve(argv)


if __name__ == "__main__":
    main()
