import json
from pathlib import Path

from datasets import Dataset

from .config import chat_data_file

data_path = Path(__file__).parent.parent / "data"


def chat_to_msg(chat: dict[str, str]):
    return {
        "messages": [
            {"role": "system", "content": "你是模仿弱智吧语气的中文助手。"},
            {"role": "user", "content": chat["instruction"]},
            {"role": "assistant", "content": chat["output"]},
        ]
    }


def get_chat_data() -> Dataset:
    with open(data_path / chat_data_file, "r") as file:
        chat_data = json.load(file)
    res = Dataset.from_list(chat_data).map(chat_to_msg)
    return res
