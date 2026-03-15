from lazyrouter.error_logger import sanitize_for_log

def test_sanitize_for_log_scalar_values():
    assert sanitize_for_log("string") == "string"
    assert sanitize_for_log(123) == 123
    assert sanitize_for_log(True) is True
    assert sanitize_for_log(None) is None

def test_sanitize_for_log_dict_non_sensitive_keys():
    input_data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    expected_data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    assert sanitize_for_log(input_data) == expected_data

def test_sanitize_for_log_dict_sensitive_keys():
    input_data = {
        "api_key": "secret-123",
        "AUTHORIZATION": "Bearer secret",
        "x-api-key": "secret",
        "X-Goog-Api-Key": "secret",
        "other_key": "visible"
    }
    expected_data = {
        "api_key": "[REDACTED]",
        "AUTHORIZATION": "[REDACTED]",
        "x-api-key": "[REDACTED]",
        "X-Goog-Api-Key": "[REDACTED]",
        "other_key": "visible"
    }
    assert sanitize_for_log(input_data) == expected_data

def test_sanitize_for_log_nested_dict():
    input_data = {
        "user": {
            "name": "Alice",
            "api_key": "secret-key"
        },
        "metadata": {
            "headers": {
                "Authorization": "Bearer token"
            }
        }
    }
    expected_data = {
        "user": {
            "name": "Alice",
            "api_key": "[REDACTED]"
        },
        "metadata": {
            "headers": {
                "Authorization": "[REDACTED]"
            }
        }
    }
    assert sanitize_for_log(input_data) == expected_data

def test_sanitize_for_log_list_of_dicts():
    input_data = [
        {"name": "test"},
        {"api_key": "secret"},
        ["nested", {"authorization": "token"}]
    ]
    expected_data = [
        {"name": "test"},
        {"api_key": "[REDACTED]"},
        ["nested", {"authorization": "[REDACTED]"}]
    ]
    assert sanitize_for_log(input_data) == expected_data

def test_sanitize_for_log_mixed_structures():
    input_data = {
        "items": [
            1,
            "string",
            {"api_key": "secret"}
        ],
        "config": {
            "x-api-key": "test"
        }
    }
    expected_data = {
        "items": [
            1,
            "string",
            {"api_key": "[REDACTED]"}
        ],
        "config": {
            "x-api-key": "[REDACTED]"
        }
    }
    assert sanitize_for_log(input_data) == expected_data
