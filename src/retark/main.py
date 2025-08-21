from typing import Literal

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from .config import MODEL_NAME
from .data_utils import get_chat_data, get_text_data


def dapt(model, text_data):
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

    trainer = SFTTrainer(
        model=model,
        train_dataset=text_data,
        args=dapt_args,
    )
    trainer.train()


def sft(model, model_name):
    chat_data = get_chat_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure EOS exists; many chat templates depend on it
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))

    # ==== 3) LoRA config (still useful on MPS to keep memory small) =============
    # If you truly want *full fine-tuning*, set peft_config=None and remove it below.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    def fmt_prompts(batch):
        texts = []
        for msgs in batch["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)
        return texts

    train_cfg = SFTConfig(
        output_dir="./sft-qwen2.5-lora-chat-mps",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        gradient_checkpointing=True,
        bf16=False,
        fp16=False, # keep False on MPS for stability
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=chat_data,
        args=train_cfg,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained("./sft-qwen2.5-lora-chat-mps")


def main():
    text_data = get_text_data()
    device = torch.device("cpu")
    use_mps = torch.backends.mps.is_available()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    dtype=torch.float16 if use_mps else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    ).to(device)

    # dapt(model, text_data)
    sft(model, MODEL_NAME)

if __name__ == "__main__":
    main()
