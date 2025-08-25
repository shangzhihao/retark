import os
import sys
from pathlib import Path
from typing import Any, Dict
import tomllib

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = PROJECT_ROOT / "config.toml"

# Load configuration from TOML file
def _load_config() -> Dict[str, Any]:
    """Load configuration from config.toml file."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)

# Load the configuration
_config = _load_config()

# Data configuration
TEXT_FILE = _config["data"]["text_file"]
CHAT_FILE = _config["data"]["chat_file"]

# Model configuration
MODEL_NAME = _config["model"]["name"]
BLOCK_SIZE = _config["model"]["block_size"]

# Dataset configuration
GROUP_DS = _config["dataset"]["group_ds"]

# LoRA configuration
LORA_R = _config["lora"]["r"]
LORA_ALPHA = _config["lora"]["alpha"]

# Environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
