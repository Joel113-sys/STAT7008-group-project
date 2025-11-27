import types
import builtins
import pytest
from unittest.mock import MagicMock, call

from agent.web_agent import WebAgent


class DummyVLLM:
    def analyze(self, **kwargs):
        return {"thought": "", "action": "DONE", "parameter": "", "completed": True}


def make_agent():
    return WebAgent(DummyVLLM())


class LocatorMock:
    def __init__(self, count_val: int, should_raise: bool = False):
        self._count_val = count_val
        self._raise = should_raise
        self.scroll_into_view_if_needed = MagicMock()
        self.click = MagicMock()

    def count(self):
        if self._raise:
            raise RuntimeError("locator error")
        return self._count_val

    @property
    def first(self):
        return self


class PageMock:
    def __init__(self, url="https://example.com"):
        self.url = url
        self._locators = {}
        self.evaluate = MagicMock()
        self.goto = MagicMock(return_value=types.SimpleNamespace(status=200))
        self.wait_for_load_state = MagicMock()

    def locator(self, sel):
        return self._locators.get(sel, LocatorMock(0))

    def set_locator(self, sel, loc):
        self._locators[sel] = loc


# ---------------- GOTO ----------------

def test_action_goto_missing_url():
    agent = make_agent()
    page = PageMock()
    assert agent._action_goto(page, "") is False


def test_action_goto_same_url_skips():
    agent = make_agent()
    page = PageMock(url="https://same.com")
    assert agent._action_goto(page, "https://same.com") is True
    page.goto.assert_not_called()


def test_action_goto_prefix_https_and_calls_safe_goto(monkeypatch):
    agent = make_agent()
    page = PageMock(url="https://other.com")
    called = {"url": None}

    def fake_safe_goto(p, url):
        called["url"] = url
        return True

    monkeypatch.setattr(agent, "_safe_goto", fake_safe_goto)
    assert agent._action_goto(page, "example.com") is True
    assert called["url"] == "https://example.com"


# ---------------- CLICK ----------------

def test_action_click_missing_descriptor():
    agent = make_agent()
    page = PageMock()
    assert agent._action_click(page, "") is False


def test_action_click_success_on_first_selector():
    agent = make_agent()
    page = PageMock()
    # The first selector is text={param}
    param = "Downloads"
    sel = f"text={param}"
    loc = LocatorMock(1)
    page.set_locator(sel, loc)

    ok = agent._action_click(page, param)
    assert ok is True
    loc.click.assert_called_once()


def test_action_click_not_found():
    agent = make_agent()
    page = PageMock()
    ok = agent._action_click(page, "Nonexistent")
    assert ok is False


# ---------------- SCROLL ----------------

def test_action_scroll_ok():
    agent = make_agent()
    page = PageMock()
    ok = agent._action_scroll(page)
    assert ok is True
    page.evaluate.assert_called_with("window.scrollBy(0, window.innerHeight)")


def test_action_scroll_exception():
    agent = make_agent()
    page = PageMock()
    page.evaluate.side_effect = RuntimeError("blocked")
    ok = agent._action_scroll(page)
    assert ok is False
