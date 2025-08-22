from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .data_utils import get_chat_ds


def sft(model, model_name):
    chat_data = get_chat_ds()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

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
        fp16=False,  # keep False on MPS for stability
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=chat_data,
        args=train_cfg,
    )

    trainer.train()
