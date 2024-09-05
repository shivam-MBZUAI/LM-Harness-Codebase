import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, logging  # type: ignore
from datasets import load_dataset
import argparse
from evaluate import load as load_evaluate  # type: ignore
from statistics import mean
from torch.cuda.amp import autocast  # type: ignore

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on the LLaMA3 model.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# Transformers logging to suppress messages
logging.set_verbosity_error()

# Step 2: Load the custom LLaMA3 model and tokenizer with mixed precision
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Utilize multiple GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# For multi-GPU usage, wrap the model in DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Set the padding token to the end-of-sequence token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model Loaded")

# Step 3: Load the dataset
dataset = load_dataset("ArkaAcharya/MMCQSD", split="train[:3]")
print("Dataset Loaded")

# Load evaluation metrics: ROUGE and BERTScore
rouge = load_evaluate("rouge")
bertscore = load_evaluate("bertscore")

# Step 4: Function to evaluate and compute ROUGE and BERTScore
def evaluate(model, tokenizer, dataset, rouge_metric, bertscore_metric):
    all_predictions = []
    all_references = []

    for id, example in enumerate(dataset):
        source_text = example["Codemixed_Question"]
        reference_text = example["summary"]
        
        # Enhanced prompt with detailed guidance on tone, structure, and content
        prompt = f"""
        You are an expert medical professional. Your task is to summarize complex medical queries in a concise, factual, and clear manner. Use no more than 100 words. The summary should prioritize critical medical information and omit irrelevant details. Ensure the tone is professional and empathetic, focusing on the patient’s symptoms, concerns, and any ongoing treatment.

        {source_text}
        """

        # Tokenize the source text directly with the enhanced prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)

        input_ids = inputs["input_ids"]

        # Clear GPU cache before generating to free up memory
        torch.cuda.empty_cache()

        # Generate summary with mixed precision and adjusted generation settings
        with autocast():
            # Access the underlying model in case of DataParallel
            if isinstance(model, torch.nn.DataParallel):
                outputs = model.module.generate(
                    input_ids=input_ids,
                    do_sample=True,  # Enable sampling for temperature and top_k/top_p
                    num_beams=5,  # Increase number of beams
                    max_new_tokens=150,  # Generate more tokens for longer summaries
                    no_repeat_ngram_size=3,  # Less restrictive repetition constraint
                    top_k=50,  # More random sampling
                    top_p=0.95,  # Increase top_p to allow for more token diversity
                    temperature=1.0,  # Remove temperature control
                    eos_token_id=tokenizer.eos_token_id  # Stop generation at EOS token
                )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    do_sample=True,  # Enable sampling for temperature and top_k/top_p
                    num_beams=5,  # Increase number of beams
                    max_new_tokens=150,  # Generate more tokens for longer summaries
                    no_repeat_ngram_size=3,  # Less restrictive repetition constraint
                    top_k=50,  # More random sampling
                    top_p=0.95,  # Increase top_p to allow for more token diversity
                    temperature=1.0,  # Remove temperature control
                    eos_token_id=tokenizer.eos_token_id  # Stop generation at EOS token
                )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Post-processing: remove common unwanted phrases like "### Input:" and "### Response:"
        prediction = prediction.replace("### Input:", "").replace("### Response:", "").strip()

        # Print initial text (source) and generated text
        print(f"Source Text: {source_text}\n")
        print(f"Source Summary: {reference_text}\n")
        print(f"Generated Summary: {prediction}\n\n")

        # Store the prediction and reference
        all_predictions.append(prediction)
        all_references.append([reference_text])

    # Compute ROUGE score
    rouge_results = rouge_metric.compute(predictions=all_predictions, references=all_references)
    
    # Compute BERTScore
    bertscore_results = bertscore_metric.compute(predictions=all_predictions, references=all_references, lang="en")

    return rouge_results, bertscore_results

# Step 5: Evaluate the model and get the ROUGE and BERTScore results
rouge_results, bertscore_results = evaluate(model, tokenizer, dataset, rouge, bertscore)

# Handling ROUGE results and formatting them correctly
if isinstance(rouge_results['rougeL'], dict):
    rouge_l_fmeasure = rouge_results['rougeL']['fmeasure']
else:
    rouge_l_fmeasure = rouge_results['rougeL']

print(f"ROUGE Score: {rouge_l_fmeasure:.2f}")

# Calculate the mean of BERTScore F1 values from the list
bertscore_f1_mean = mean(bertscore_results['f1'])
print(f"BERTScore (F1): {bertscore_f1_mean:.2f}")

print("Evaluation Complete")
