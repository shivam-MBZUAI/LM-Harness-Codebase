import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from datasets import load_dataset
import argparse
from sklearn.metrics import f1_score

# Transformers logging to suppress messages
logging.set_verbosity_error()

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on the LLaMA3 model for MMLU task.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# Clear CUDA cache
torch.cuda.empty_cache()

# Print model path
print(f"Evaluating model from: {args.model_name}")

# Load the dataset from Hugging Face
dataset = load_dataset("shuyuej/Hindi-MMLU-Medical-Genetics-Benchmark", split="test")

# Load the LLaMA3 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Add a padding token if it's not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.resize_token_embeddings(len(tokenizer))  # Update the model's token embeddings to account for the new pad token

# Function to encode the inputs for the model
def preprocess_function(examples):
    # Concatenate the context with each possible answer
    questions = examples['question']
    choices = list(zip(examples['A'], examples['B'], examples['C'], examples['D']))
    
    # Tokenize the inputs and calculate the logits for each choice
    tokenized = []
    for question, choice_set in zip(questions, choices):
        encoded_choices = []
        for choice in choice_set:
            encoded = tokenizer(question, choice, truncation=True, padding=True, return_tensors="pt")
            encoded_choices.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
        tokenized.append(encoded_choices)
    
    return tokenized

# Preprocess the dataset
encoded_dataset = preprocess_function(dataset)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Mapping from label string to integer index
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# Evaluation function
def evaluate_model(encoded_dataset, model):
    model.eval()
    total, correct = 0, 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, choices in enumerate(encoded_dataset):
            logits = []
            for choice in choices:
                input_ids = choice['input_ids'].unsqueeze(0).to(device)
                attention_mask = choice['attention_mask'].unsqueeze(0).to(device)
                
                # Get the logits for the final token for each choice
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logit = outputs.logits[:, -1, :]  # We take the logit of the last token
                
                # Select the logit corresponding to the last input token of each choice
                logits.append(logit.squeeze(0)[input_ids[0, -1]].item())
            
            # Convert logits to tensor and determine the choice with the highest logit
            logits = torch.tensor(logits)
            predictions = torch.argmax(logits).item()

            # Convert label from string ('A', 'B', 'C', 'D') to integer
            labels = label_mapping[dataset['label'][i]]

            # Collect predictions and labels for F1 score calculation
            all_predictions.append(predictions)
            all_labels.append(labels)
            
            # Print debugging information for the first 5 questions
            if i < 5:
                print(f"Question {i}:")
                print(f"Logits: {logits}")
                print(f"Prediction: {predictions}")
                print(f"Actual label: {dataset['label'][i]}")

            # Compare the prediction with the actual label
            correct += (predictions == labels)
            total += 1
    
    # Calculate Accuracy
    accuracy = correct / total

    # Calculate F1 Score (macro-average, assuming multiclass classification)
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return accuracy, f1

# Evaluate the model on the dataset
accuracy, f1 = evaluate_model(encoded_dataset, model)
print(f"Accuracy of the LLaMA3 model on the MMLU task: {accuracy:.2f}")
print(f"F1 Score of the LLaMA3 model on the MMLU task: {f1:.2f}")
