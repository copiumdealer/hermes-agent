#!/usr/bin/env python3
"""Resolve and probe a Railway public URL without embedding the raw .app URL in the command line.

Usage:
  python3 /opt/hermes/scripts/railway_public_url_check.py
  python3 /opt/hermes/scripts/railway_public_url_check.py --path /health --path /docs

Resolution order:
1. PINTEREST_PUBLISHER_PUBLIC_BASE_URL / RAILWAY_PUBLIC_URL / HERMES_DEPLOY_URL
2. *_FILE variants for the names above
3. /opt/data/railway_public_url.txt
4. ~/.config/hermes/railway_public_url.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

URL_ENV_VARS = (
    "PINTEREST_PUBLISHER_PUBLIC_BASE_URL",
    "RAILWAY_PUBLIC_URL",
    "HERMES_DEPLOY_URL",
)
FILE_ENV_VARS = (
    "PINTEREST_PUBLISHER_PUBLIC_BASE_URL_FILE",
    "RAILWAY_PUBLIC_URL_FILE",
    "HERMES_DEPLOY_URL_FILE",
)
DEFAULT_FILES = (
    Path("/opt/data/railway_public_url.txt"),
    Path.home() / ".config" / "hermes" / "railway_public_url.txt",
)
DEFAULT_PATHS = (
    "/api/pinterest-publisher/health",
    "/publisher/",
)


def _normalize(raw: str | None) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "https://" + raw.lstrip("/")
    return raw.rstrip("/")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def resolve_base_url() -> tuple[str | None, str | None]:
    for name in URL_ENV_VARS:
        value = _normalize(os.getenv(name))
        if value:
            return value, f"env:{name}"

    for name in FILE_ENV_VARS:
        file_path = os.getenv(name, "").strip()
        if not file_path:
            continue
        value = _normalize(_read_text(Path(file_path).expanduser()))
        if value:
            return value, f"file-env:{name}"

    for path in DEFAULT_FILES:
        value = _normalize(_read_text(path))
        if value:
            return value, f"file:{path}"

    return None, None


def probe(base_url: str, paths: Iterable[str]) -> int:
    print(f"Base URL: {base_url}")
    exit_code = 0
    for raw_path in paths:
        path = raw_path if raw_path.startswith("/") else "/" + raw_path
        url = urljoin(base_url + "/", path.lstrip("/"))
        request = Request(
            url,
            headers={
                "User-Agent": "Hermes-Railway-URL-Check/1.0",
                "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read(400).decode("utf-8", errors="replace").strip()
                print(f"- {path}: {response.status} {url}")
                if body:
                    print(f"  {body}")
        except HTTPError as exc:
            body = exc.read(400).decode("utf-8", errors="replace").strip()
            print(f"- {path}: {exc.code} {url}")
            if body:
                print(f"  {body}")
            exit_code = 1
        except URLError as exc:
            print(f"- {path}: ERROR {url}")
            print(f"  {exc}")
            exit_code = 1
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", dest="paths", help="Path to probe; may be passed multiple times")
    parser.add_argument("--json", action="store_true", help="Print resolved base URL metadata as JSON and exit")
    args = parser.parse_args()

    base_url, source = resolve_base_url()
    if not base_url:
        print(
            "No Railway public URL found. Set one of PINTEREST_PUBLISHER_PUBLIC_BASE_URL / RAILWAY_PUBLIC_URL / HERMES_DEPLOY_URL,\n"
            "or store it in /opt/data/railway_public_url.txt",
            file=sys.stderr,
        )
        return 2

    if args.json:
        print(json.dumps({"base_url": base_url, "source": source}, ensure_ascii=False))
        return 0

    return probe(base_url, args.paths or DEFAULT_PATHS)


if __name__ == "__main__":
    raise SystemExit(main())
