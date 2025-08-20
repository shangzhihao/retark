from typing import Literal

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

from .config import MODEL_NAME
from .data_utils import get_chat_data, get_text_data


def main():
    text_data = get_text_data()
    chat_data = get_chat_data()

    device: Literal["mps", "cpu"] = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    # tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    dapt_args = TrainingArguments(
        output_dir="out_dapt",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=1000,
    )

    trainer_dapt = SFTTrainer(
        model=model,
        train_dataset=text_data,
        args=dapt_args,
    )
    trainer_dapt.train()


if __name__ == "__main__":
    main()
