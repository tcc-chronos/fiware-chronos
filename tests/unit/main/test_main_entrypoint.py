from __future__ import annotations

import runpy


def test_main_module_invokes_worker(monkeypatch):
    executed = {}

    def fake_main() -> None:
        executed["called"] = True

    monkeypatch.setattr("src.main.worker.main", fake_main)

    runpy.run_module("src.main.__main__", run_name="__main__")

    assert executed["called"] is True
