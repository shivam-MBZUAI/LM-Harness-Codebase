import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, logging # type: ignore
from datasets import load_dataset
import argparse
from evaluate import load as load_evaluate  # type: ignore

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on the LLaMA3 model.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# Transformers logging to suppress message
logging.set_verbosity_error()

# Step 2: Load the custom LLaMA3 model and tokenizer with mixed precision
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)

# Utilize multiple GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

# Set the padding token to the end-of-sequence token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model Loaded")

# Step 3: Load the dataset (adjust the sample size if needed)
dataset = load_dataset("Cognitive-Lab/GoogleIndicGenBench_flores_enxx_in", name="hi", split="test[:3]")
print("Dataset Loaded")

# Load the BLEU metric for evaluation
bleu = load_evaluate("sacrebleu")

# Step 4: Function to translate and evaluate
def evaluate(model, tokenizer, dataset, metric):
    all_predictions = []
    all_references = []

    for id, example in enumerate(dataset):
        source_text = example["source"]  # Source language (English)
        reference_text = example["target"]  # Reference translation (Hindi)
        prompt = f"Translate the following English text to Hindi:\n{source_text}"

        # Tokenize the source text with the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)

        # Extract the input_ids tensor
        input_ids = inputs["input_ids"]

        # Generate the translation
        if torch.cuda.device_count() > 1:
            outputs = model.module.generate(
                input_ids=input_ids,
                num_beams=6,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256
            )
        else:
            outputs = model.generate(
                input_ids=input_ids,
                num_beams=6,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256
            )

        # Decode the generated output
        prediction = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Remove the prompt part from the prediction
        translated_text = prediction.replace(f"Translate the following English text to Hindi:\n{source_text}", "").strip()

        # Store the prediction and reference
        all_predictions.append(translated_text)
        all_references.append([reference_text])

        if id < 3:
            print(f"Processed Example {id}")
            print("Generated Translation:")
            print(translated_text)
            print("\nOriginal Reference:")
            print(reference_text)

    # Compute BLEU score
    results = metric.compute(predictions=all_predictions, references=all_references)
    return results


# Step 5: Evaluate the model
results = evaluate(model, tokenizer, dataset, bleu)

# Print the BLEU score
print(f"BLEU Score: {results['score']:.2f}")

print("Evaluation Complete")
