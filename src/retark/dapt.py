from peft import LoraConfig, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from .data_utils import tokenize, group_texts, tokenizer
from .config import LORA_R
from datasets import Dataset


def dapt(model: PreTrainedModel, text_data: Dataset, group=False) -> Trainer:
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R*2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    peft_model = get_peft_model(model, peft_cfg)
    tokenized = tokenize(text_data)
    if group:
        train_ds = tokenized.map(group_texts, batched=True)
    else:
        train_ds = tokenized

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        # FIXIT: remove hard coding
        output_dir="out_dapt",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=500,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        # processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    return trainer
