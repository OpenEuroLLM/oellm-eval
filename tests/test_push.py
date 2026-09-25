"""`oellm-eval push` and `collect --push`, against a fake dashboard on localhost."""

import inspect
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from oellm import push
from oellm.push import PushError, push_envelope, push_path, push_results

ENVELOPE = {
    "version": "1.2",
    "generated_at": "2026-09-21T10:00:00+00:00",
    "runs": [{"submitted_by": "ivan", "limit": None}],
    "results": [
        {
            "model": "m",
            "task": "copa",
            "n_shot": 0,
            "metric": "acc,none",
            "performance": 0.5,
        }
    ],
}


class FakeDashboard:
    """Answers POSTs from a queue of (status, body); records what it received."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[tuple[int, dict, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                status, payload, headers = (
                    outer.responses.pop(0)
                    if outer.responses
                    else (200, {"status": "ingested", "rows": 1}, {})
                )
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(
                    payload.encode()
                    if isinstance(payload, str)
                    else json.dumps(payload).encode()
                )

            do_GET = do_POST

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def dashboard():
    server = FakeDashboard()
    yield server
    server.close()


@pytest.fixture
def envelope(tmp_path):
    path = tmp_path / "run" / "eval_results.json"
    path.parent.mkdir()
    path.write_text(json.dumps(ENVELOPE, indent=2))
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("OELLM_DASH_URL", "OELLM_DASH_TOKEN", "OELLM_DASH_TOKEN_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


class TestRequest:
    def test_sends_the_envelope_with_token_and_origin(self, dashboard, envelope):
        outcome = push_envelope(envelope, dashboard.url, "edt_secret")
        assert (outcome.status, outcome.rows) == ("ingested", 1)
        (req,) = dashboard.requests
        assert req["path"] == "/api/ingest"
        assert json.loads(req["body"]) == ENVELOPE
        assert req["headers"]["Authorization"] == "Bearer edt_secret"
        assert req["headers"]["X-Elliot-Source"].endswith(str(envelope.resolve()))
        assert req["headers"]["User-Agent"].startswith("oellm-eval/")

    def test_prefix_is_kept_in_the_request_path(self, dashboard, envelope, monkeypatch):
        """The hosted dashboard lives under /elliot-dashboard behind a proxy."""
        monkeypatch.setenv("OELLM_DASH_TOKEN", "edt_x")
        push_path(envelope, server=dashboard.url + "/elliot-dashboard/")
        assert dashboard.requests[0]["path"] == "/elliot-dashboard/api/ingest"

    def test_duplicate_counts_as_success(self, dashboard, envelope):
        dashboard.responses.append((200, {"status": "duplicate", "rows": 0}, {}))
        outcome = push_envelope(envelope, dashboard.url, "t")
        assert outcome.status == "duplicate" and outcome.ok


class TestFailures:
    def test_server_errors_are_retried_with_backoff(self, dashboard, envelope):
        dashboard.responses += [(500, {}, {}), (503, {}, {})]
        waits: list[float] = []
        outcome = push_envelope(envelope, dashboard.url, "t", sleep=waits.append)
        assert outcome.status == "ingested"
        assert len(dashboard.requests) == 3 and waits == [1, 2]

    def test_gives_up_after_the_last_retry(self, envelope):
        waits: list[float] = []
        outcome = push_envelope(
            envelope, "http://127.0.0.1:9", "t", timeout=2, sleep=waits.append
        )
        assert outcome.status == "failed" and not outcome.ok and outcome.unreachable
        assert waits == [1, 2, 4] and "gave up" in outcome.detail

    @pytest.mark.parametrize(
        "reply",
        [[1, 2], {"ok": True}, "<html>sign in</html>"],
        ids=["json-array", "json-object-without-status", "html-page"],
    )
    def test_a_reply_not_from_the_dashboard_fails(self, dashboard, envelope, reply):
        dashboard.responses.append((200, reply, {}))
        waits: list[float] = []
        outcome = push_envelope(envelope, dashboard.url, "t", sleep=waits.append)
        assert outcome.status == "failed" and "dashboard" in outcome.detail
        assert len(dashboard.requests) == 1 and waits == []

    def test_a_timeout_is_not_retried(self, envelope):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        accepted: list[socket.socket] = []

        def accept_forever():
            try:
                while True:
                    accepted.append(srv.accept()[0])
            except OSError:
                pass

        threading.Thread(target=accept_forever, daemon=True).start()
        waits: list[float] = []
        outcome = push_envelope(
            envelope,
            f"http://127.0.0.1:{srv.getsockname()[1]}",
            "t",
            timeout=1,
            sleep=waits.append,
        )
        srv.close()
        for conn in accepted:
            conn.close()
        assert outcome.status == "failed" and outcome.unreachable
        assert len(accepted) == 1 and waits == []

    def test_an_unreachable_dashboard_stops_the_remaining_pushes(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("OELLM_DASH_TOKEN", "t")
        monkeypatch.setattr(push, "BACKOFF_S", ())
        for name in ("a", "b", "c"):
            d = tmp_path / "out" / name
            d.mkdir(parents=True)
            (d / "eval_results.json").write_text(json.dumps(ENVELOPE))
        outcomes = push_path(tmp_path / "out", server="http://127.0.0.1:9")
        assert [o.status for o in outcomes] == ["failed"] * 3
        assert outcomes[0].unreachable
        assert all("not sent" in o.detail for o in outcomes[1:])

    def test_client_errors_are_final(self, dashboard, envelope):
        dashboard.responses.append((401, {"detail": "missing or invalid token"}, {}))
        waits: list[float] = []
        outcome = push_envelope(envelope, dashboard.url, "bad", sleep=waits.append)
        assert outcome.status == "failed" and "token" in outcome.detail
        assert len(dashboard.requests) == 1 and waits == []

    def test_validation_error_from_the_dashboard_is_shown(self, dashboard, envelope):
        dashboard.responses.append((422, {"detail": "runs must be a list"}, {}))
        assert "runs must be a list" in push_envelope(envelope, dashboard.url, "t").detail

    def test_redirects_are_not_followed(self, dashboard, envelope):
        """urllib would resend the token to the redirect target."""
        dashboard.responses.append((302, {}, {"Location": dashboard.url + "/elsewhere"}))
        outcome = push_envelope(
            envelope, dashboard.url, "edt_secret", sleep=lambda s: None
        )
        assert outcome.status == "failed" and "redirected" in outcome.detail
        assert [r["path"] for r in dashboard.requests] == ["/api/ingest"]

    def test_non_envelope_file_is_skipped_not_sent(self, dashboard, tmp_path):
        other = tmp_path / "results.json"
        other.write_text(json.dumps({"results": {"copa": {"acc": 0.5}}}))
        assert push_envelope(other, dashboard.url, "t").status == "skipped"
        assert dashboard.requests == []


class TestConfiguration:
    def test_plain_http_to_a_remote_host_is_refused(self):
        with pytest.raises(PushError, match="plain http"):
            push.resolve_server("http://dashboard.example.org/elliot")
        assert push.resolve_server("https://dashboard.example.org/elliot/") == (
            "https://dashboard.example.org/elliot"
        )

    def test_a_malformed_port_is_a_configuration_error(self):
        with pytest.raises(PushError, match="not a valid dashboard address"):
            push.resolve_server("https://dashboard.example.org:abc/elliot")

    def test_missing_address_is_a_clear_error(self):
        with pytest.raises(PushError, match="OELLM_DASH_URL"):
            push.resolve_server(None)

    def test_token_sources_in_order(self, tmp_path, monkeypatch):
        default = Path(tmp_path / "home/.config/oellm/dash_token")
        default.parent.mkdir(parents=True)
        default.write_text("from-default\n")
        default.chmod(0o600)
        assert push.resolve_token(None) == "from-default"

        monkeypatch.setenv("OELLM_DASH_TOKEN", "from-env")
        assert push.resolve_token(None) == "from-env"

        env_file = tmp_path / "env_token"
        env_file.write_text("from-env-file")
        env_file.chmod(0o600)
        monkeypatch.setenv("OELLM_DASH_TOKEN_FILE", str(env_file))
        assert push.resolve_token(None) == "from-env-file"

        flag_file = tmp_path / "flag_token"
        flag_file.write_text("  from-flag  \n")
        flag_file.chmod(0o600)
        assert push.resolve_token(str(flag_file)) == "from-flag"

    def test_missing_token_is_a_clear_error(self, tmp_path):
        with pytest.raises(PushError, match="dash_token"):
            push.resolve_token(None)
        with pytest.raises(PushError, match="not found"):
            push.resolve_token(str(tmp_path / "nope"))

    def test_the_token_can_never_be_passed_on_the_command_line(self):
        """argv is visible in `ps` and shell history."""
        assert "token" not in inspect.signature(push_results).parameters
        assert set(inspect.signature(push_results).parameters) == {
            "path",
            "server",
            "token_file",
            "dry_run",
            "verbose",
        }


class TestCommand:
    def test_directory_is_searched_recursively(self, dashboard, tmp_path, monkeypatch):
        monkeypatch.setenv("OELLM_DASH_TOKEN", "t")
        for name in ("a", "b/nested"):
            d = tmp_path / "out" / name
            d.mkdir(parents=True)
            (d / "eval_results.json").write_text(json.dumps(ENVELOPE))
        (tmp_path / "out" / "a" / "provenance.json").write_text("{}")
        outcomes = push_path(tmp_path / "out", server=dashboard.url)
        assert [o.status for o in outcomes] == ["ingested", "ingested"]
        assert len(dashboard.requests) == 2

    def test_dry_run_sends_nothing_and_needs_no_token(self, dashboard, envelope):
        (outcome,) = push_path(envelope, server=dashboard.url, dry_run=True)
        assert (outcome.status, outcome.rows) == ("dry-run", 1)
        assert dashboard.requests == []

    def test_a_file_that_is_not_an_envelope_fails_the_command(
        self, dashboard, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("OELLM_DASH_TOKEN", "t")
        csv = tmp_path / "eval_results.csv"
        csv.write_text("model_name,task\nm,copa\n")
        with pytest.raises(SystemExit) as failed:
            push_results(str(csv), server=dashboard.url)
        assert failed.value.code == 1 and dashboard.requests == []

    def test_nothing_to_push_is_an_error(self, dashboard, tmp_path):
        with pytest.raises(PushError, match="collect"):
            push_path(tmp_path, server=dashboard.url)

    def test_exit_codes(self, dashboard, envelope, monkeypatch):
        monkeypatch.setenv("OELLM_DASH_TOKEN", "t")
        push_results(str(envelope), server=dashboard.url)  # success: returns

        dashboard.responses.append((401, {}, {}))
        with pytest.raises(SystemExit) as failed:
            push_results(str(envelope), server=dashboard.url)
        assert failed.value.code == 1

        with pytest.raises(SystemExit) as misconfigured:
            push_results(str(envelope), server=None)
        assert misconfigured.value.code == 2

    def test_registered_as_a_cli_command(self):
        from oellm.main import app

        names = {c.name or c.callback.__name__ for c in app.registered_commands}
        assert "push" in names


LM_EVAL_RESULT = {
    "model_name": "EleutherAI/pythia-70m",
    "results": {"copa": {"acc,none": 0.5, "alias": "copa"}},
    "configs": {"copa": {"num_fewshot": 0}},
}


class TestCollectPush:
    def _run_dir(self, tmp_path: Path) -> Path:
        results = tmp_path / "run" / "results"
        results.mkdir(parents=True)
        (results / "abc.json").write_text(json.dumps(LM_EVAL_RESULT))
        return tmp_path / "run"

    def test_collect_push_publishes_what_it_wrote(self, dashboard, tmp_path, monkeypatch):
        from oellm.results import collect_results

        monkeypatch.setenv("OELLM_DASH_URL", dashboard.url)
        monkeypatch.setenv("OELLM_DASH_TOKEN", "t")
        run = self._run_dir(tmp_path)
        collect_results(str(run), str(run / "eval_results.csv"), push=True)

        (req,) = dashboard.requests
        sent = json.loads(req["body"])
        assert sent == json.loads((run / "eval_results.json").read_text())
        assert sent["results"][0]["task"] == "copa"

    def test_a_failed_push_never_fails_collect(self, tmp_path, monkeypatch):
        from oellm.results import collect_results

        monkeypatch.setenv("OELLM_DASH_URL", "https://dashboard.invalid")
        monkeypatch.setattr(push, "BACKOFF_S", ())
        run = self._run_dir(tmp_path)
        collect_results(str(run), str(run / "eval_results.csv"), push=True)  # no token
        assert (run / "eval_results.csv").exists() and (
            run / "eval_results.json"
        ).exists()

    def test_without_the_flag_nothing_is_sent(self, dashboard, tmp_path, monkeypatch):
        from oellm.results import collect_results

        monkeypatch.setenv("OELLM_DASH_URL", dashboard.url)
        monkeypatch.setenv("OELLM_DASH_TOKEN", "t")
        run = self._run_dir(tmp_path)
        collect_results(str(run), str(run / "eval_results.csv"))
        assert dashboard.requests == []
