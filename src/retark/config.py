from pathlib import Path

TEXT_FILE = "ruozhiba-post-annual.json"
CHAT_FILE = "ruozhiba_qa.json"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_SIZE = 256
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
GROUP_DS = False