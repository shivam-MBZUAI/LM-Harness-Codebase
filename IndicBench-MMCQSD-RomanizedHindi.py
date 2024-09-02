import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
import argparse
from evaluate import load as load_evaluate
from torch.cuda.amp import autocast
from statistics import mean

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on the LLaMA3 model.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# Transformers logging to suppress messages
logging.set_verbosity_error()

# Step 2: Load the custom LLaMA3 model and tokenizer with mixed precision
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)

# Utilize multiple GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the padding token to the end-of-sequence token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model Loaded")

# Step 3: Load the dataset (adjust the sample size if needed)
dataset = load_dataset("ArkaAcharya/MMCQSD", split="train[:1000]")  # Increased to 100 samples
print("Dataset Loaded")

# Load evaluation metrics
bleu = load_evaluate("sacrebleu")
rouge = load_evaluate("rouge")
bertscore = load_evaluate("bertscore")

# Step 4: Function to translate and evaluate
def evaluate(model, tokenizer, dataset, bleu_metric, rouge_metric, bertscore_metric):
    all_predictions = []
    all_references = []

    for id, example in enumerate(dataset):
        source_text = example["Codemixed_Question"]  # Codemixed question text
        reference_text = example["summary"]  # Summary text
        prompt = f"Summarize the following Code-Mixed Question to English:\n{source_text}"

        # Tokenize the source text directly without any additional prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=3500).to(device)

        # Extract the input_ids tensor
        input_ids = inputs["input_ids"]

        # Generate the summary with adjusted parameters to reduce repetition
        with autocast():
            outputs = model.generate(
                input_ids=input_ids,
                num_beams=8,
                max_new_tokens=450,
                no_repeat_ngram_size=3,  # Prevents repetition of 3-grams
                top_k=30,  # Controls diversity
                top_p=0.90,  # Nucleus sampling
                temperature=0.5  # Controls randomness
            )

        # Decode only the generated new tokens (excluding the original input)
        generated_tokens = outputs[0][input_ids.shape[-1]:]  # Outputs is a tensor, so index it directly
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Store the prediction and reference
        all_predictions.append(prediction)
        all_references.append([reference_text])
    
    # Compute BLEU score
    bleu_results = bleu_metric.compute(predictions=all_predictions, references=all_references)
    
    # Compute ROUGE score
    rouge_results = rouge_metric.compute(predictions=all_predictions, references=all_references)
    
    # Compute BERTScore
    bertscore_results = bertscore_metric.compute(predictions=all_predictions, references=all_references, lang="en")

    return bleu_results, rouge_results, bertscore_results

# Step 5: Evaluate the model
bleu_results, rouge_results, bertscore_results = evaluate(model, tokenizer, dataset, bleu, rouge, bertscore)

# Print the evaluation results
print(f"BLEU Score: {bleu_results['score']:.2f}")

# Handling ROUGE results and formatting them correctly
if isinstance(rouge_results['rougeL'], dict):
    rouge_l_fmeasure = rouge_results['rougeL']['fmeasure']
else:
    rouge_l_fmeasure = rouge_results['rougeL']

print(f"ROUGE-L Score: {rouge_l_fmeasure:.2f}")

# Calculate the mean of BERTScore F1 values from the list
bertscore_f1_mean = mean(bertscore_results['f1'])
print(f"BERTScore (F1): {bertscore_f1_mean:.2f}")

print("Evaluation Complete")
