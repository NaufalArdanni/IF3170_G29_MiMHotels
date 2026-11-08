"""Smoke tests for the package skeleton."""

import ronaldohotel


def test_package_has_docstring() -> None:
    """The top-level package imports and documents itself."""
    assert ronaldohotel.__doc__
