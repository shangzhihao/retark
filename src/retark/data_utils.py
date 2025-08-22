import json
import re

from datasets import Dataset
from transformers import AutoTokenizer

from .config import (BLOCK_SIZE, CHAT_FILE, DATA_DIR, HF_TOKEN, MODEL_NAME,
                     TEXT_FILE)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=True, token=HF_TOKEN)


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


def tokenize_fun(sample):
    return tokenizer(
        # FIXIT: text is a magic string
        sample["text"],
        truncation=True,
        max_length=BLOCK_SIZE,
        return_special_tokens_mask=True,
    )


def tokenize(ds):
    tokenized = ds.map(
        tokenize_fun,
        batched=True,
        remove_columns=ds.column_names,
    )
    return tokenized


def group_texts(samples):
    concatenated = {k: sum(samples[k], []) for k in samples.keys()}
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    grouped = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    grouped["labels"] = grouped["input_ids"].copy()
    return grouped


def get_chat_ds() -> Dataset:
    with open(DATA_DIR / CHAT_FILE) as file:
        chat_data = json.load(file)
    res = Dataset.from_list(chat_data).map(chat_to_msg)
    return res


def get_text_ds() -> Dataset:
    with open(DATA_DIR / TEXT_FILE) as file:
        text_data = json.load(file)
    text_data = [{"content": s["content"]} for s in text_data]
    res = Dataset.from_list(text_data).map(text_to_msg)
    return res
