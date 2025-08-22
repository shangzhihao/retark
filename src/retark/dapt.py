

from peft import LoraConfig, get_peft_model
from transformers import (DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from .data_utils import tokenize, group_texts, tokenizer


def dapt(model, text_data, group=False):
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    tokenized = tokenize(text_data)
    if group:
        train_ds = tokenized.map(group_texts, batched=True)
    else:
        train_ds = tokenized

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
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
        model=model,
        args=training_args,
        train_dataset=train_ds,
        # processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()