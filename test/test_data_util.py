# test/test_data_utils.py
import json

import pytest

# Import the module under test
import retark.data_utils as du


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """
    Create a temporary data directory and redirect retark.data_utils to use it.
    We keep the filenames from retark.config via the bound names in data_utils.
    """
    # Make a temp data dir
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Point the module's DATA_DIR to our temp dir
    monkeypatch.setattr(du, "DATA_DIR", data_dir, raising=True)

    return data_dir


def test_get_chat_data(tmp_data_dir):
    # Prepare minimal chat-style JSON (list of {instruction, output})
    chat_payload = [
        {"instruction": "只剩一个心脏了还能活吗？", "output": "能，人本来就只有一个心脏。"},
        {"instruction": "天气如何？", "output": "阴天，别出门。"},
    ]

    # Respect the filename the module expects (from config bound at import)
    chat_file = tmp_data_dir / du.CHAT_FILE
    chat_file.write_text(json.dumps(chat_payload, ensure_ascii=False), encoding="utf-8")

    ds = du.get_chat_data()

    # Basic shape check
    assert len(ds) == 2

    # Check the message structure of the first example
    first = ds[0]["messages"]
    # Should be: system, user, assistant
    assert isinstance(first, list) and len(first) == 3
    assert first[0]["role"] == "system" and "弱智吧" in first[0]["content"]
    assert first[1]["role"] == "user" and first[1]["content"] == chat_payload[0]["instruction"]
    assert first[2]["role"] == "assistant" and first[2]["content"] == chat_payload[0]["output"]


def test_get_text_data(tmp_data_dir):
    # Prepare numbered text items (list of {content})
    text_payload = [
        {"content": "1. First text"},
        {"content": "23, Another text"},
        {"content": "305、中文内容"},
        {"content": "42 This is spaced"},
    ]

    text_file = tmp_data_dir / du.TEXT_FILE
    text_file.write_text(json.dumps(text_payload, ensure_ascii=False), encoding="utf-8")

    ds = du.get_text_data()

    # Should extract the text part (after the leading number+separator)
    got = ds["text"]
    assert got == ["First text", "Another text", "中文内容", "This is spaced"]
