import torch # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, logging # type: ignore
import argparse
from sklearn.metrics import accuracy_score # type: ignore
import pandas as pd
import json

# Constants for the Llama3-Instruct template
BEGIN_TEXT = "<|begin_of_text|>"
START_HEADER_USER = "<|start_header_id|>user<|end_header_id|>"
START_HEADER_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>"
EOT = "<|eot_id|>"

# Transformers logging to suppress messages
logging.set_verbosity_error()

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on multiple models for MMLU task.")
parser.add_argument("--model_names", nargs='+', required=True, help="List of model directory paths.")
parser.add_argument("--dataset_path", type=str, default="Datasets/SafetyData.json", help="Path to the dataset JSON file.")
parser.add_argument("--output_csv", type=str, default="Datasets/model_predictions.csv", help="Path to the output CSV file.")
parser.add_argument("--f1_output_csv", type=str, default="Datasets/model_f1_scores.csv", help="Path to the output F1 score CSV file.")
args = parser.parse_args()

# Clear CUDA cache
torch.cuda.empty_cache()

# Load the dataset from the local JSON file
with open(args.dataset_path, 'r') as f:
    dataset = json.load(f)

# Function to encode the inputs with the Llama3-Instruct template
def apply_llama_template(examples, tokenizer):
    tokenized = []
    for example in examples:
        # Apply Llama3-Instruct template
        question = f"{BEGIN_TEXT}{START_HEADER_USER}{example['instruction']}{EOT}{START_HEADER_ASSISTANT}"
        choices = [
            f"{question}{example['option_a']}{EOT}",
            f"{question}{example['option_b']}{EOT}",
            f"{question}{example['option_c']}{EOT}",
            f"{question}{example['option_d']}{EOT}",
        ]
        
        # Tokenize the inputs with the template
        encoded_choices = []
        for choice in choices:
            encoded = tokenizer(choice, truncation=True, padding=True, return_tensors="pt")
            encoded_choices.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
        tokenized.append(encoded_choices)
    
    return tokenized

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping from label string to integer index
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# Evaluate multiple models
def evaluate_models(model_names, dataset):
    # Initialize the rows with questions and correct answers
    csv_data = []
    f1_data = []  # To store F1 scores
    for example in dataset:
        row = {
            "Question": example['instruction'],
            "Correct Answer": example[f'option_{example["answer"].lower()}']
        }
        csv_data.append(row)

    # Initialize the prediction columns for each model
    for row in csv_data:
        for model_name in model_names:
            row[model_name] = ""  # Placeholder for model predictions

    # Iterate over each model
    for model_name in model_names:
        print(f"Evaluating model: {model_name}")
        
        # Load the model and tokenizer once for all questions
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # Add a padding token if it's not present
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))  # Update the token embeddings to account for pad token

        # Preprocess the dataset for the current model with the Llama3-Instruct template
        encoded_dataset = apply_llama_template(dataset, tokenizer)

        model_predictions = []
        correct_labels = []

        # Iterate over each example for prediction
        for i, example in enumerate(dataset):
            # Predict for the current example
            logits = []
            with torch.no_grad():
                for choice in encoded_dataset[i]:
                    input_ids = choice['input_ids'].unsqueeze(0).to(device)
                    attention_mask = choice['attention_mask'].unsqueeze(0).to(device)

                    # Get the logits for the final token for each choice
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logit = outputs.logits[:, -1, :]  # We take the logit of the last token
                    logits.append(logit.squeeze(0)[input_ids[0, -1]].item())

            # Convert logits to tensor and determine the choice with the highest logit
            logits = torch.tensor(logits)
            predictions = torch.argmax(logits).item()
            predicted_answer = example[f'option_{chr(predictions + 97)}']  # Get the predicted answer ('a', 'b', 'c', 'd')

            # Add the model prediction to the corresponding row
            csv_data[i][model_name] = predicted_answer

            # Collect predictions and labels for F1 score calculation
            model_predictions.append(predictions)
            correct_label = label_mapping[example["answer"]]
            correct_labels.append(correct_label)

        # After all questions, calculate F1 score for this model
        f1 = accuracy_score(correct_labels, model_predictions)
        f1_data.append({"Model": model_name, "F1 Score": f1})

        # Clear the model and empty the cache to free up GPU memory
        del model
        torch.cuda.empty_cache()

    # Save CSV file with predictions
    df = pd.DataFrame(csv_data)
    df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

    # Save CSV file with F1 scores
    f1_df = pd.DataFrame(f1_data)
    f1_df.to_csv(args.f1_output_csv, index=False)
    print(f"F1 scores saved to {args.f1_output_csv}")

# Run the evaluation for multiple models
evaluate_models(args.model_names, dataset)
