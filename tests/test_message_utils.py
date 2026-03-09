import pytest
from lazyrouter.message_utils import content_to_text

@pytest.mark.parametrize(
    "content, expected",
    [
        (None, ""),
        ("hello", "hello"),
        (
            [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
            "hello\nworld",
        ),
        (
            [
                "hello",
                "world",
            ],
            "hello\nworld",
        ),
        (
            [
                {"type": "text", "text": "hello"},
                "world",
            ],
            "hello\nworld",
        ),
        (
            [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}},
                "world",
            ],
            "hello\nworld",
        ),
        (
            [
                {"type": "text"},
                {"type": "text", "text": 123}, # invalid text val should skip
                "hello",
            ],
            "hello",
        ),
        (
            [
                {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}},
            ],
            "",
        ),
        (
            {"type": "text", "text": "hello"},
            "hello",
        ),
        (
            {"type": "text", "text": 123}, # invalid text val
            "",
        ),
        (
            {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}},
            "",
        ),
        (
            123,
            "123",
        ),
        (
            [],
            "",
        ),
        (
            {},
            "",
        ),
    ],
)
def test_content_to_text(content, expected):
    assert content_to_text(content) == expected
