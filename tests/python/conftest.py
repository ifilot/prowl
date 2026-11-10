import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def runs_dir() -> Path:
    """Private VASP fixtures generated with tools/run_vasp_fixture.py (never committed).

    Set PROWL_TEST_RUNS to the runs directory to enable the real-data tests."""
    d = os.environ.get("PROWL_TEST_RUNS")
    if not d or not Path(d).is_dir():
        pytest.skip("real VASP fixtures not available (set PROWL_TEST_RUNS); synthetic tests still run")
    return Path(d)
