import hashlib

from lazyrouter.message_utils import content_to_text
from lazyrouter.session_utils import extract_session_key


def _make_request():
    from unittest.mock import MagicMock
    req = MagicMock()
    req.model_extra = {}
    return req


def test_auto_fallback_returns_hash_of_first_user_message():
    req = _make_request()
    messages = [{"role": "user", "content": "hello world"}]
    result = extract_session_key(req, messages)
    text = content_to_text("hello world").strip()
    expected = "auto:" + hashlib.sha256(text.encode()).hexdigest()[:16]
    assert result == expected


def test_auto_fallback_returns_none_for_empty_user_content():
    req = _make_request()
    messages = [{"role": "user", "content": "   "}]
    assert extract_session_key(req, messages) is None


def test_auto_fallback_returns_none_when_no_user_message():
    req = _make_request()
    messages = [{"role": "assistant", "content": "hi"}]
    assert extract_session_key(req, messages) is None
