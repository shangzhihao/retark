from transformers import PreTrainedModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from .data_utils import tokenizer
def sft(model: PreTrainedModel, chat_data: Dataset):

    peft_model = model
    train_cfg = SFTConfig(
        output_dir="./out/sft-lora-chat",
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
        model=peft_model,
        train_dataset=chat_data,
        args=train_cfg,
        processing_class=tokenizer,
    )

    trainer.train()
    return trainer
