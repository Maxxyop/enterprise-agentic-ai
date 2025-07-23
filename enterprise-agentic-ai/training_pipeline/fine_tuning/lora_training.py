# lora_training.py

import torch
from transformers import LoraModel, LoraTrainer, LoraTrainingArguments

def fine_tune_lora(model_name, train_dataset, eval_dataset, output_dir, epochs=3, learning_rate=5e-5):
    # Load the model
    model = LoraModel.from_pretrained(model_name)

    # Set training arguments
    training_args = LoraTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
    )

    # Initialize the trainer
    trainer = LoraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    # Example usage
    model_name = "deepseek_r1"
    train_dataset = None  # Load your training dataset here
    eval_dataset = None   # Load your evaluation dataset here
    output_dir = "./lora_model_output"

    fine_tune_lora(model_name, train_dataset, eval_dataset, output_dir)