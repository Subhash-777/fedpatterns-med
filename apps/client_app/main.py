"""
apps/client_app/main.py — Stable re-export of the active baseline's ClientApp.

To switch baselines, update the import below. Do NOT change pyproject.toml.
"""

from fedpatterns.core.app.client_app import app  # M2 Patternized active baseline

__all__ = ["app"]
