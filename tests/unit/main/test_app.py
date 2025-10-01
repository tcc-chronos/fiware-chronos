from __future__ import annotations

import pytest

from src.main import app as module_app
from src.main.app import create_app


class _StubMongoDatabase:
    async def create_indexes(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_create_app_initializes_lifespan(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.main.container.MongoDatabase",
        lambda *args, **kwargs: _StubMongoDatabase(),
    )

    app = create_app()
    assert app.title

    async with app.router.lifespan_context(app):
        assert app.state.started_at is not None

    # Ensure module-level app is instantiated
    assert isinstance(module_app.app, type(app))
