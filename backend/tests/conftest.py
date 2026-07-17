"""Repo-wide pytest fixtures.

SEMARK_CORPUS_RULES_PATH must never be set while running the test suite: it
points at a gitignored, org-specific ruleset (see
app.pipeline.corpus_rules.RULES_PATH_ENV_VAR), and because corpus_rules
caches the loaded ruleset in a process-global (_cached_rules), a leaked
value would silently poison every golden/quality test that expects the
bundled defaults for the rest of the process -- only the *first* call to
get_rules() actually reads the env var.

This fixture is session-scoped and autouse so the var is stripped and the
cache is reset before ANY test module imports/calls get_rules(), including
modules with no idea this concern exists. It is intentionally compatible
with tests/test_corpus_rules.py's own function-scoped `_fresh_rules_cache`
autouse fixture: that fixture uses `monkeypatch.delenv` (which restores
whatever value was present before each test function -- "unset", since this
fixture unsets it once for the whole session) and calls the same
`reset_rules_cache()`. The two layer cleanly: function-scoped resets on top
of a session-scoped baseline. (A session-scoped fixture cannot depend on
`monkeypatch`, which pytest only provides at function scope, hence the
plain `os.environ.pop` here instead.)
"""

import os

import pytest

from app.pipeline.corpus_rules import RULES_PATH_ENV_VAR, reset_rules_cache


@pytest.fixture(autouse=True, scope="session")
def _no_corpus_rules_override():
    os.environ.pop(RULES_PATH_ENV_VAR, None)
    reset_rules_cache()
    yield
    os.environ.pop(RULES_PATH_ENV_VAR, None)
    reset_rules_cache()
