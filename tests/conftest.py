"""Shared pytest configuration and fixtures for psyduck tests."""

import matplotlib

matplotlib.use("Agg")

import pytest

from psyduck import Spin


@pytest.fixture
def spin_half():
    return Spin(I=1 / 2)


@pytest.fixture
def spin_sb():
    """Antimony-like I = 7/2 spin in the default state."""
    return Spin(I=7 / 2)


@pytest.fixture(params=[1 / 2, 1, 3 / 2, 5 / 2, 7 / 2], ids=lambda I: f"I={I}")
def spin_any(request):
    return Spin(I=request.param)
