import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, logging  # type: ignore
from datasets import load_dataset
import argparse
from evaluate import load as load_evaluate  # type: ignore
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
        You are an expert medical professional. Your task is to summarize complex medical queries in a concise, factual, and clear manner. Use no more than 100 words. The summary should prioritize critical medical information and omit irrelevant details. Ensure the tone is professional and empathetic, focusing on the patient’s symptoms, concerns, and any ongoing treatment. The summary should be coherent and easy for both medical professionals and concerned family members to understand. If applicable, reference potential diagnoses or treatments in a general way without specific medical recommendations.

        Example 1:
        Source: "Namaste doctor, mere 4 saal ke bhatije ko autisim hai aur char din pehle uske right eye suj gaya tha, jiske baad infection ho gaya aur uske eye ke aas-pass sore jaise dikhne lage. Ab ye infection left eye mein bhi fail raha hai. Bachon ke hospital ke doctors ko is bare mein kuch bhi pata nahi hai, lekin herpes ko bilkul bhi nahi dismiss kiya gaya hai. Uska image neeche attach kiya gaya hai. Tabiyat ki test results discharge par ya toh inconclusive the ya phir tayyar nahi the, waise bhi 48 ghante se zyada ho gaye hain. Ek doctor keh raha tha ki ye sthiti andha-bhakti bhi kar sakti hai. Wo teen alag-alag prakar ke IV antibiotics par hain, lekin ab toh lagta hai ki uski sthiti uske face ke dusre hisse mein bhi fail rahi hai. Uske autisim ki wajah se hum ye nahi samajh sakte ki usko dard ya takleef hai ya nahi. Humein usko suffer karte dekhna dil ko tod deta hai. Agar aap hume koi jankari de sakein toh hum aapki ati abhari honge."
        Summary: "A four-year-old boy with autism has developed an infection in his right eye that has now spread to his left. The infection is worsening despite IV antibiotics, and there are concerns about potential blindness. Herpes has not been ruled out, but test results remain inconclusive. The boy's autism prevents him from expressing pain, leaving his family deeply concerned. Doctors are unsure of the underlying cause, and the infection seems to be spreading to other areas of his face."

        Example 2:
        Source: "Hello doctor, Mujhe lagbhag ek mahine se taklif hai. Mujhe antibiotics le rahe hain, lekin ye theek nahi ho raha hai. Mujhe throat mein thoda dard hota hai. Kabhi kabhi kaan mein dull pain bhi hota hai. Kya ye tonsil ya throat cancer ke koi lakshan hai? Uska image attached hai neeche."
        Summary: "The patient has been experiencing persistent throat pain for almost a month, which has not improved with antibiotics. Additionally, there is occasional dull ear pain. They are concerned that the symptoms might be related to tonsillitis or throat cancer. An image has been provided for further examination."

        Example 3:
        Source: "Namaste doctor, Haal hi mein maine ek open lymph node biopsy ke saath lymph node extraction karwaya hai. General anesthesia se uthne ke baad se hi mujhe pata chala hai ki mera trapezius muscle kaam nahi kar raha hai aur mere kaan mein sunapan hai. Mere doctor, jo ki ek qualified ENT specialist hai, ne mujhe bataya hai ki ye naasamjh nerve damage nahi hai aur mujhe bas swelling kam hone ka wait karna chahiye. Meri surgery ki chot 3 cm lambi hai mere kaan ke niche. Mere surgery ke 4 din baad, meri chot achhe se bhar rahi hai, lekin trapezius kaam karne ka koi sign bilkul nahi hai. Mere A and E mein neck ka ultrasound hua hai aur chot mein koi hematoma ya fluid nahi hai. Mujhe Emergency Room doctor ne Neiromidin ki prescribing ki hai. Maine aapke reference ke liye apni tasveer attach ki hai."
        Summary: "Following a lymph node biopsy and extraction, the patient is experiencing trapezius muscle paralysis and ear numbness. The ENT specialist believes this is due to temporary swelling post-surgery. The wound is healing well, with no fluid buildup or hematoma detected via ultrasound. The patient has been prescribed Neiromidin to aid in recovery. An image has been attached for further reference."

        Now summarize the following query based on the provided examples in no more than 100 words:

        {source_text}
        """

        # Tokenize the source text directly with the enhanced prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1500).to(device)

        input_ids = inputs["input_ids"]

        # Generate summary with tighter generation parameters and EOS token control
        outputs = model.generate(
            input_ids=input_ids,
            num_beams=5,
            max_new_tokens=100,  # Limit the length for concise summaries
            no_repeat_ngram_size=5,  # Prevents repetition of 5-grams
            top_k=20,  # Reduce randomness
            top_p=0.80,  # Reduce randomness
            temperature=0.6,  # Control randomness
            eos_token_id=tokenizer.eos_token_id  # Stop generation at EOS token
        )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Post-processing: remove common unwanted endings
        if "Here is an extract" in prediction:
            prediction = prediction.split("Here is an extract")[0].strip()

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
