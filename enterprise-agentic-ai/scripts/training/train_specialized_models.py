import os
import sys
import json
import logging
from ai_agents.specialized_models.pentest_gpt.training import train_model
from training_pipeline.data_collection.dataset_builder import build_dataset

logging.basicConfig(level=logging.INFO)

def main():
    # Step 1: Build the dataset
    logging.info("Building dataset for specialized models...")
    dataset = build_dataset()
    
    # Step 2: Train the Pentest GPT model
    logging.info("Starting training for Pentest GPT model...")
    model_path = train_model(dataset)
    
    # Step 3: Save the trained model
    if model_path:
        logging.info(f"Model trained successfully and saved at: {model_path}")
    else:
        logging.error("Model training failed.")

if __name__ == "__main__":
    main()