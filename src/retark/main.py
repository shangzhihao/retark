
from transformers import AutoModelForCausalLM
from .data_utils import get_text_ds, get_chat_ds, tokenizer
from .config import HF_TOKEN, MODEL_NAME, GROUP_DS
from .dapt import dapt
from .sft import sft

def main():
    text_data = get_text_ds()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto",
        device_map="auto", token=HF_TOKEN
    )

    dapt_trainer = dapt(model, text_data, group=GROUP_DS)
    dapt_trainer.save_model("./out/dapt_out")
    
    chat_data = get_chat_ds()
    model = AutoModelForCausalLM.from_pretrained("./out/dapt_out")
    
    sft_trainer = sft(model, chat_data)
    sft_trainer.save_model("./out/sft_out")


if __name__ == "__main__":
    main()
