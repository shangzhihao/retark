import os
from pathlib import Path

TEXT_FILE = "post_mini.json"
CHAT_FILE = "qa_mini.json"
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
BLOCK_SIZE = 256
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
GROUP_DS = False
HF_TOKEN = os.environ.get("HF_TOKEN")
LORA_R = 4
LORA_ALPHA = 16
