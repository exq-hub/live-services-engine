# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the HTTP routes, run without the app's startup.

The TestClient is used without its context manager, so the lifespan
(model loading, collection loading, audit log) never runs. Each test
stubs only what its route reads.
"""

import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from app import __version__


@pytest.fixture
def client():
    return TestClient(main.app)


def _model_manager(embedding_type, names):
    return SimpleNamespace(
        embedding_type=embedding_type, loaded_model_names=lambda: names
    )


class TestVersion:
    def test_version_comes_from_pyproject(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject.open("rb") as f:
            expected = tomllib.load(f)["project"]["version"]

        assert __version__ == expected

    def test_root_reports_the_version(self, client):
        assert client.get("/").json()["version"] == __version__


class TestHealth:
    def test_reports_version_and_loaded_models_per_family(self, client, monkeypatch):
        fake_container = SimpleNamespace(
            config_manager=SimpleNamespace(
                config=SimpleNamespace(collections=["testcol"])
            ),
            database_repository=MagicMock(),
            index_repository=MagicMock(),
            clip_model_manager=_model_manager("CLIP", ["ViT-B-32"]),
            text_model_manager=_model_manager("Text", []),
        )
        monkeypatch.setattr(main, "container", fake_container)

        body = client.get("/health").json()

        assert body["status"] == "healthy"
        assert body["version"] == __version__
        assert body["models_loaded"] == {"CLIP": ["ViT-B-32"], "Text": []}
