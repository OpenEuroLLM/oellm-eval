"""Publish collected results to an ELLIOT dashboard (``oellm-eval push``).

Runs on cluster login nodes, which have outbound HTTPS but accept no inbound
connections: results are pushed out, the dashboard never reaches in. Standard
library only, so it works in whatever environment already runs ``oellm-eval``.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ENVELOPE_NAME = "eval_results.json"
DEFAULT_TOKEN_FILE = "~/.config/oellm/dash_token"
TIMEOUT_S = 30
BACKOFF_S = (1, 2, 4)  # one entry per retry
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class PushError(Exception):
    """A problem the user has to fix (configuration, not a transient failure)."""


@dataclass(frozen=True)
class PushOutcome:
    path: Path
    status: str  # ingested | duplicate | dry-run | skipped | failed
    rows: int = 0
    detail: str = ""
    unreachable: bool = False

    @property
    def ok(self) -> bool:
        return self.status not in ("failed", "skipped")


def resolve_server(server: str | None) -> str:
    url = (server or os.environ.get("OELLM_DASH_URL") or "").strip().rstrip("/")
    if not url:
        raise PushError("no dashboard address: pass --server or set OELLM_DASH_URL")
    parts = urllib.parse.urlsplit(url)
    try:
        valid = parts.scheme in ("http", "https") and bool(parts.hostname)
        valid = valid and parts.port != 0
    except ValueError:  # malformed port
        valid = False
    if not valid:
        raise PushError(f"not a valid dashboard address: {url!r}")
    if parts.scheme == "http" and parts.hostname not in _LOCAL_HOSTS:
        raise PushError(
            f"refusing to send a token over plain http to {parts.hostname}; use https"
        )
    return url


def resolve_token(token_file: str | None) -> str:
    """--token-file, $OELLM_DASH_TOKEN_FILE, $OELLM_DASH_TOKEN, then the default
    file. There is no option taking the token itself: command lines show up in
    ``ps`` and shell history."""
    explicit = token_file or os.environ.get("OELLM_DASH_TOKEN_FILE")
    if explicit:
        return _read_token_file(Path(explicit).expanduser(), must_exist=True)
    if os.environ.get("OELLM_DASH_TOKEN", "").strip():
        return os.environ["OELLM_DASH_TOKEN"].strip()
    token = _read_token_file(Path(DEFAULT_TOKEN_FILE).expanduser(), must_exist=False)
    if not token:
        raise PushError(
            "no dashboard token: put it in ~/.config/oellm/dash_token (chmod 600), "
            "or set OELLM_DASH_TOKEN_FILE or OELLM_DASH_TOKEN"
        )
    return token


def _read_token_file(path: Path, *, must_exist: bool) -> str:
    if not path.is_file():
        if must_exist:
            raise PushError(f"token file not found: {path}")
        return ""
    if path.stat().st_mode & 0o077:
        logging.warning(f"{path} is readable by other users; run: chmod 600 {path}")
    token = path.read_text().strip()
    if not token and must_exist:
        raise PushError(f"token file is empty: {path}")
    return token


def find_envelopes(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(p.rglob(ENVELOPE_NAME))
    if p.is_file():
        return [p]
    raise PushError(f"no such file or directory: {p}")


def _load_envelope(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib would replay the Authorization header to wherever a redirect
    points, and turn the POST into a GET. Neither is acceptable here."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, f"redirected to {newurl}", headers, fp
        )


def _source_label(path: Path) -> str:
    label = f"{socket.gethostname()}:{path.resolve()}"
    return label.encode("ascii", "replace").decode()


def push_envelope(
    path: Path,
    server: str,
    token: str,
    *,
    timeout: float = TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> PushOutcome:
    """POST one envelope. Retries connection errors and 5xx; a 4xx is final."""
    data = _load_envelope(path)
    if data is None:
        return PushOutcome(path, "skipped", detail="not an eval_results envelope")

    from oellm import __version__

    request = urllib.request.Request(
        f"{server}/api/ingest",
        data=json.dumps(data).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Elliot-Source": _source_label(path),
            "User-Agent": f"oellm-eval/{__version__}",
        },
    )
    opener = urllib.request.build_opener(_NoRedirect)
    error, unreachable = "", False
    for attempt in range(len(BACKOFF_S) + 1):
        if attempt:
            sleep(BACKOFF_S[attempt - 1])
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as e:
            error, unreachable = f"HTTP {e.code}: {_error_detail(e)}", False
            if e.code < 500:
                return PushOutcome(path, "failed", detail=error)
            continue
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, TimeoutError):
                # Not retried: every retry would wait the full timeout again.
                return PushOutcome(
                    path,
                    "failed",
                    detail=f"no answer from the dashboard within {timeout:g} s",
                    unreachable=True,
                )
            error, unreachable = f"{type(e).__name__}: {reason}", True
            continue
        return _parse_reply(path, raw)
    return PushOutcome(
        path,
        "failed",
        detail=f"{error} (gave up after {len(BACKOFF_S)} retries)",
        unreachable=unreachable,
    )


def _parse_reply(path: Path, raw: bytes) -> PushOutcome:
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    if not isinstance(body, dict) or body.get("status") not in ("ingested", "duplicate"):
        return PushOutcome(
            path,
            "failed",
            detail="the reply did not come from an ELLIOT dashboard; check the address",
        )
    rows = body.get("rows")
    return PushOutcome(path, body["status"], rows if isinstance(rows, int) else 0)


def _error_detail(e: urllib.error.HTTPError) -> str:
    if e.code == 401:
        return "the dashboard rejected the token (wrong, or revoked)"
    if 300 <= e.code < 400:
        return f"{e.reason}; set the address to the final https location"
    try:
        return str(json.loads(e.read()).get("detail", e.reason))
    except (ValueError, AttributeError, OSError):
        return str(e.reason)


def push_path(
    path: str | Path,
    *,
    server: str | None = None,
    token_file: str | None = None,
    dry_run: bool = False,
) -> list[PushOutcome]:
    url = resolve_server(server)
    envelopes = find_envelopes(path)
    if not envelopes:
        raise PushError(
            f"no {ENVELOPE_NAME} under {path}; collect writes it to the directory "
            "it is run from, or next to --output-csv"
        )
    if dry_run:
        outcomes = []
        for f in envelopes:
            data = _load_envelope(f)
            outcomes.append(
                PushOutcome(f, "dry-run", len(data["results"]))
                if data
                else PushOutcome(f, "skipped", detail="not an eval_results envelope")
            )
        return outcomes
    token = resolve_token(token_file)
    outcomes = []
    for i, f in enumerate(envelopes):
        outcome = push_envelope(f, url, token)
        outcomes.append(outcome)
        if outcome.unreachable:
            outcomes += [
                PushOutcome(g, "failed", detail="not sent: the dashboard is unreachable")
                for g in envelopes[i + 1 :]
            ]
            break
    return outcomes


def _report(outcomes: list[PushOutcome], server: str) -> None:
    for o in outcomes:
        if o.status == "ingested":
            logging.info(f"pushed {o.path}: {o.rows} rows")
        elif o.status == "duplicate":
            logging.info(f"already on the dashboard: {o.path}")
        elif o.status == "dry-run":
            logging.info(f"would push {o.path}: {o.rows} rows to {server}")
        elif o.status == "skipped":
            logging.error(f"skipped {o.path}: {o.detail}")
        else:
            logging.error(f"failed {o.path}: {o.detail}")


def push_results(
    path: str,
    *,
    server: str | None = None,
    token_file: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """
    Push collected results to the ELLIOT dashboard.

    Run it on the login node after `collect`. Pushing the same results twice is
    harmless: the dashboard recognises them and changes nothing.

    Args:
        path: An eval_results.json file, or a directory searched recursively for them
        server: Dashboard address, e.g. https://host/elliot-dashboard (default: $OELLM_DASH_URL)
        token_file: File holding your upload token (default: $OELLM_DASH_TOKEN_FILE,
            then $OELLM_DASH_TOKEN, then ~/.config/oellm/dash_token)
        dry_run: Show what would be pushed; sends nothing and needs no token
        verbose: Enable verbose logging
    """
    from oellm.utils import _setup_logging

    _setup_logging(verbose)
    try:
        outcomes = push_path(path, server=server, token_file=token_file, dry_run=dry_run)
    except PushError as e:
        logging.error(str(e))
        raise SystemExit(2) from None
    _report(outcomes, resolve_server(server))
    if not all(o.ok for o in outcomes):
        raise SystemExit(1)


def push_after_collect(envelope: Path) -> bool:
    """``collect --push``: publish the envelope just written. A push problem
    must never cost the user their collected results, so nothing is raised."""
    try:
        outcomes = push_path(envelope)
        _report(outcomes, resolve_server(None))
        ok = all(o.ok for o in outcomes)
    except Exception as e:
        logging.warning(f"push failed: {e}")
        ok = False
    if not ok:
        logging.warning(
            f"results are saved locally; retry with: oellm-eval push {envelope}"
        )
    return ok
