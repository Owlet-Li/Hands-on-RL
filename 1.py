trainer = Trainer(
    model=base_model,
    train_dataset=instruction_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=3
    ),
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)
trainer.train()

