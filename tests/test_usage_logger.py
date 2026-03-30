from unittest.mock import patch

import pytest
from lazyrouter.usage_logger import estimate_tokens, DEFAULT_TOKEN_MODEL


def test_estimate_tokens_empty_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


@patch("lazyrouter.usage_logger.litellm.token_counter")
def test_estimate_tokens_default_model(mock_token_counter):
    mock_token_counter.return_value = 5
    result = estimate_tokens("hello world")

    assert result == 5
    mock_token_counter.assert_called_once_with(model=DEFAULT_TOKEN_MODEL, text="hello world")


@patch("lazyrouter.usage_logger.litellm.token_counter")
def test_estimate_tokens_explicit_model(mock_token_counter):
    mock_token_counter.return_value = 10
    result = estimate_tokens("hello world", model="claude-3-opus")

    assert result == 10
    mock_token_counter.assert_called_once_with(model="claude-3-opus", text="hello world")


@patch("lazyrouter.usage_logger.litellm.token_counter")
def test_estimate_tokens_fallback_on_exception(mock_token_counter):
    # Setup mock to fail on the first call (claude-3-opus) and succeed on the second (gpt-4)
    def side_effect(model, text, **kwargs):
        if model == "claude-3-opus":
            raise ValueError("Unsupported model")
        return 7

    mock_token_counter.side_effect = side_effect

    result = estimate_tokens("hello world", model="claude-3-opus")

    assert result == 7
    assert mock_token_counter.call_count == 2
    mock_token_counter.assert_any_call(model="claude-3-opus", text="hello world")
    mock_token_counter.assert_any_call(model=DEFAULT_TOKEN_MODEL, text="hello world")


@patch("lazyrouter.usage_logger.litellm.token_counter")
def test_estimate_tokens_raises_if_default_model_fails(mock_token_counter):
    mock_token_counter.side_effect = ValueError("Default model failed")

    with pytest.raises(ValueError, match="Default model failed"):
        estimate_tokens("hello world")

    # If the user provides a different model but fallback also fails
    mock_token_counter.reset_mock()
    mock_token_counter.side_effect = ValueError("Fallback failed")

    with pytest.raises(ValueError, match="Fallback failed"):
        estimate_tokens("hello world", model="claude-3-opus")
