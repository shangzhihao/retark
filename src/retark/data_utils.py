import json
import re
from pathlib import Path

from datasets import Dataset

from .config import CHAT_FILE, TEXT_FILE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def chat_to_msg(chat: dict[str, str]) -> dict[str, list]:
    return {
        "messages": [
            {"role": "system", "content": "你是模仿弱智吧语气的中文助手。"},
            {"role": "user", "content": chat["instruction"]},
            {"role": "assistant", "content": chat["output"]},
        ]
    }


def text_to_msg(text: dict[str, str]) -> dict[str, str]:
    pattern = re.compile(r"^\s*\d{1,3}[\.,、 ]\s*(.*)$")
    m = pattern.match(text["content"])
    if m:
        return {"text": m.group(1)}
    else:
        return {"text": text["content"]}


def get_chat_data() -> Dataset:
    with open(DATA_DIR / CHAT_FILE) as file:
        chat_data = json.load(file)
    res = Dataset.from_list(chat_data).map(chat_to_msg)
    return res


def get_text_data() -> Dataset:
    with open(DATA_DIR / TEXT_FILE) as file:
        text_data = json.load(file)
    res = Dataset.from_list(text_data).map(text_to_msg)
    return res
