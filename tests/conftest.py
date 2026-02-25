import os

import pytest


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    # Make CLI tests deterministic
    monkeypatch.setenv("TESTING_FLAG", "true")
    monkeypatch.setenv("JOB_FETCHER_RUN_NUMBER", "3")
