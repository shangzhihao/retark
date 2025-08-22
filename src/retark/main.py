
from transformers import AutoModelForCausalLM
from .data_utils import get_text_ds
from .config import MODEL_NAME, GROUP_DS
from .dapt import dapt


def main():
    text_data = get_text_ds()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )
    dapt(model, text_data, group=GROUP_DS)

if __name__ == "__main__":
    main()
