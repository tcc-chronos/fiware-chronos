from __future__ import annotations

import logging
import os

import pytest

from src.shared.env import load_secret_file_variables


def _unset(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    monkeypatch.delenv(key, raising=False)


def test_load_secret_file_variables_reads_content(tmp_path, monkeypatch):
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("s3cr3t\n", encoding="utf-8")

    monkeypatch.setenv("APP_SECRET_FILE", str(secret_file))
    _unset(monkeypatch, "APP_SECRET")

    load_secret_file_variables()

    assert os.environ["APP_SECRET"] == "s3cr3t"


def test_load_secret_file_variables_logs_missing_file(monkeypatch, caplog):
    missing_path = "/tmp/does-not-exist"
    monkeypatch.setenv("MISSING_SECRET_FILE", missing_path)
    _unset(monkeypatch, "MISSING_SECRET")

    with caplog.at_level(logging.WARNING):
        load_secret_file_variables()

    assert any(record.message == "env.secret_file.missing" for record in caplog.records)


def test_load_secret_file_variables_handles_decode_error(tmp_path, monkeypatch, caplog):
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b"\xff\xfe\xfd")

    monkeypatch.setenv("BINARY_SECRET_FILE", str(binary_file))
    _unset(monkeypatch, "BINARY_SECRET")

    with caplog.at_level(logging.WARNING):
        load_secret_file_variables()

    assert any(
        record.message == "env.secret_file.decode_failed" for record in caplog.records
    )


def test_load_secret_file_variables_handles_os_error(monkeypatch, caplog):
    def _raise_os_error(self, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setenv("BROKEN_SECRET_FILE", "/tmp/any")
    _unset(monkeypatch, "BROKEN_SECRET")
    monkeypatch.setattr("src.shared.env.Path.read_text", _raise_os_error, raising=False)

    with caplog.at_level(logging.WARNING):
        load_secret_file_variables()

    assert any(
        record.message == "env.secret_file.load_failed" for record in caplog.records
    )


def test_load_secret_file_variables_skips_existing_target(monkeypatch):
    monkeypatch.setenv("EXISTING_SECRET", "present")
    monkeypatch.setenv("EXISTING_SECRET_FILE", "/tmp/ignored")

    load_secret_file_variables()

    assert os.environ["EXISTING_SECRET"] == "present"


def test_load_secret_file_variables_skips_empty_path(monkeypatch):
    _unset(monkeypatch, "EMPTY_SECRET")
    monkeypatch.setenv("EMPTY_SECRET_FILE", "")

    load_secret_file_variables()

    assert "EMPTY_SECRET" not in os.environ
