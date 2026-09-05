"""Tests for the pytest-subunit plugin."""
import io
import os
import subprocess
import sys

import pytest
import pytest_subunit as _pytest_subunit_module
from subunit.v2 import ByteStreamToStreamResult
from testtools import StreamResult

pytest_plugins = ["pytester"]

_PLUGIN_DIR = os.path.dirname(os.path.abspath(_pytest_subunit_module.__file__))


class RecordingResult(StreamResult):
    """Collect subunit status events for later inspection."""

    def __init__(self):
        super().__init__()
        self.events = []

    def status(self, **kwargs):
        self.events.append(kwargs)


def parse_subunit(stream_bytes):
    """Parse a subunit v2 byte stream into recorded events."""
    result = RecordingResult()
    parser = ByteStreamToStreamResult(
        io.BytesIO(stream_bytes), non_subunit_name="stdout"
    )
    parser.run(result)
    return result.events


def statuses_for(events, test_id):
    return [e["test_status"] for e in events if e.get("test_id") == test_id]


def run_pytest(pytester, *args):
    """Run pytest in a subprocess, returning (returncode, stdout_bytes, stderr_bytes).

    pytester rewrites HOME so the plugin installed under ~/.local becomes
    invisible to a subprocess — propagate PYTHONPATH so the plugin can still be
    imported.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        _PLUGIN_DIR + os.pathsep + existing if existing else _PLUGIN_DIR
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "pytest_subunit",
            *args,
        ],
        cwd=str(pytester.path),
        capture_output=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_passing_test_emits_success(pytester):
    pytester.makepyfile(
        """
        def test_ok():
            assert 1 == 1
        """
    )
    returncode, stdout, stderr = run_pytest(pytester, "--subunit")
    assert returncode == 0, f"stderr={stderr!r} stdout={stdout!r}"
    events = parse_subunit(stdout)
    test_id = "test_passing_test_emits_success.py::test_ok"
    statuses = statuses_for(events, test_id)
    assert "exists" in statuses
    assert "inprogress" in statuses
    assert "success" in statuses
    assert "fail" not in statuses


def test_failing_test_emits_fail(pytester):
    pytester.makepyfile(
        """
        def test_bad():
            assert 1 == 2
        """
    )
    returncode, stdout, stderr = run_pytest(pytester, "--subunit")
    # plugin forces exit code 0
    assert returncode == 0
    events = parse_subunit(stdout)
    test_id = "test_failing_test_emits_fail.py::test_bad"
    statuses = statuses_for(events, test_id)
    assert "fail" in statuses
    assert "success" not in statuses


def test_skipped_test_emits_skip(pytester):
    pytester.makepyfile(
        """
        import pytest
        @pytest.mark.skip(reason="nope")
        def test_skip_me():
            pass
        """
    )
    returncode, stdout, stderr = run_pytest(pytester, "--subunit")
    assert returncode == 0
    events = parse_subunit(stdout)
    test_id = "test_skipped_test_emits_skip.py::test_skip_me"
    statuses = statuses_for(events, test_id)
    assert "skip" in statuses
    assert "success" not in statuses


def test_xfail_test_emits_xfail(pytester):
    pytester.makepyfile(
        """
        import pytest
        @pytest.mark.xfail
        def test_expected_to_fail():
            assert False
        """
    )
    returncode, stdout, stderr = run_pytest(pytester, "--subunit")
    assert returncode == 0
    events = parse_subunit(stdout)
    test_id = "test_xfail_test_emits_xfail.py::test_expected_to_fail"
    statuses = statuses_for(events, test_id)
    assert "xfail" in statuses


def test_uxsuccess_test_emits_uxsuccess(pytester):
    pytester.makepyfile(
        """
        import pytest
        @pytest.mark.xfail(strict=True)
        def test_unexpected_pass():
            assert True
        """
    )
    returncode, stdout, stderr = run_pytest(pytester, "--subunit")
    assert returncode == 0
    events = parse_subunit(stdout)
    test_id = "test_uxsuccess_test_emits_uxsuccess.py::test_unexpected_pass"
    statuses = statuses_for(events, test_id)
    assert "uxsuccess" in statuses


def test_exit_status_is_zero_even_on_failure(pytester):
    pytester.makepyfile(
        """
        def test_will_fail():
            assert False
        """
    )
    returncode, _, stderr = run_pytest(pytester, "--subunit")
    assert returncode == 0


def test_no_subunit_flag_uses_default_reporter(pytester):
    pytester.makepyfile(
        """
        def test_ok():
            assert True
        """
    )
    returncode, stdout, stderr = run_pytest(pytester)
    assert returncode == 0
    assert b"1 passed" in stdout


def test_subunit_load_list_filters_items(pytester, tmp_path):
    pytester.makepyfile(
        test_load_list="""
        def test_one():
            pass
        def test_two():
            pass
        def test_three():
            pass
        """
    )
    load_list = tmp_path / "tests.list"
    load_list.write_text(
        "test_load_list.py::test_one\ntest_load_list.py::test_three\n"
    )
    returncode, stdout, stderr = run_pytest(
        pytester, "--subunit", "--subunit-load-list", str(load_list)
    )
    assert returncode == 0
    events = parse_subunit(stdout)
    ids_seen = {e["test_id"] for e in events if e.get("test_id")}
    assert "test_load_list.py::test_one" in ids_seen
    assert "test_load_list.py::test_three" in ids_seen
    assert "test_load_list.py::test_two" not in ids_seen


def test_collectonly_emits_exists_events(pytester):
    pytester.makepyfile(
        """
        def test_a():
            pass
        def test_b():
            pass
        """
    )
    returncode, stdout, stderr = run_pytest(pytester, "--subunit", "--collect-only")
    assert returncode == 0
    events = parse_subunit(stdout)
    exists_ids = {
        e["test_id"] for e in events if e.get("test_status") == "exists"
    }
    assert "test_collectonly_emits_exists_events.py::test_a" in exists_ids
    assert "test_collectonly_emits_exists_events.py::test_b" in exists_ids


def test_utc_tzinfo_returns_zero_offset():
    import datetime

    from pytest_subunit import utc

    now = datetime.datetime.now(utc)
    assert utc.utcoffset(now) == datetime.timedelta(0)
    assert utc.dst(now) == datetime.timedelta(0)
    assert utc.tzname(now) == "UTC"
