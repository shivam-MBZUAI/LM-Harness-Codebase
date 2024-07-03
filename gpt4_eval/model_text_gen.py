import os
import json
import sys
import argparse
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from utils import TASKS, VICUNA, SELF_INSTRUCT, SUMMARIZATION, SAFETY, FRESHQA, IQEVAL, ITC, QUANT, AR, EN
from utils import get_response_out_file_path, vicuna_80_ques, self_instruct_seeds, summarization_ar, summarization_en, safety_gen, freshqa_path, iqeval_path, itc_path, quant_path, gpt4_prompts_file, gen_prompts_file, read_jsonl, write_jsonl, gpt4_gen


device = "cuda"


class ModelOutput:
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        if self.args.gpt_gen:
            self.model_name = "gpt4"
        else:
            self.model_name = args.model_name
        self.task = args.task
        self.lang = args.lang
        if args.seq2seq:
            self.AUTO_MODEL_CLASS = AutoModelForSeq2SeqLM
        else:
            self.AUTO_MODEL_CLASS = AutoModelForCausalLM

        self.input_file = self.get_input_file()
        if self.args.cross:
            self.output_file = get_response_out_file_path(self.model_name, self.task, "cross")
        else:
            self.output_file = get_response_out_file_path(self.model_name, self.task, self.lang)

        self.prompt = self.get_prompt()

        self.tokenizer = None
        self.model = None

        self.input_keys = {
            VICUNA: 'question_ar' if self.lang == AR else 'question',
            SELF_INSTRUCT: 'question_ar' if self.lang == AR else 'question',
            SUMMARIZATION: 'question_ar' if self.lang == AR else 'question',
            SAFETY: "question",
            IQEVAL: 'instruction',
            FRESHQA: 'instruction',
            ITC: 'instruction',
            QUANT: 'instruction',
        }

        print("**********")
        print(f"model name: {self.model_name}")
        print(f"model path: {self.model_path}")
        print(f"lang: {self.lang}")
        print(f"task: {self.task}")
        print(f"input file: {self.input_file}")
        print(f"output file: {self.output_file}")
        print(f"prompt used: {self.prompt}")

    def get_input_file(self):
        if self.args.input_file != '':
            return self.args.input_file
        task_to_input_file = {
            VICUNA: vicuna_80_ques,
            SELF_INSTRUCT: self_instruct_seeds,
            SUMMARIZATION: {AR: summarization_ar, EN: summarization_en},
            SAFETY: safety_gen,
            IQEVAL: iqeval_path,
            FRESHQA: freshqa_path,
            ITC: itc_path,
            QUANT: quant_path,
        }
        if self.task == SUMMARIZATION:
            return task_to_input_file[self.task][self.lang]
        else:
            return task_to_input_file[self.task]

    def get_prompt(self):
        if self.args.gpt_gen:
            with open(gpt4_prompts_file) as f:
                prompts = json.load(f)
            return prompts[self.lang]["gpt4_gen"]

        with open(gen_prompts_file) as f:
            prompts = json.load(f)
        task_to_prompt = {
            VICUNA: "deployment_prompt",
            SELF_INSTRUCT: "preai_prompt",
            SUMMARIZATION: "summ_prompt",
            SAFETY: "no_prompt",
            IQEVAL: "deployment_prompt",
            FRESHQA: "deployment_prompt",
            ITC: "deployment_prompt",
            QUANT: "deployment_prompt",
        }
        prompt_key = task_to_prompt[self.task]
        return prompts[self.lang][prompt_key]

    def load_tokenizer(self):
        print(f"Loading tokenizer from {self.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print("tokenizer loaded.")
        return tokenizer

    def load_model(self):
        print(f"Loading model from {self.model_path}...")
        model = self.AUTO_MODEL_CLASS.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        print("model loaded.")
        model.eval()
        return model

    def gen_text(self, text):
        if text == '':
            return ''

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        input_len = input_ids.shape[-1]
        if "llama3" in self.model_name:
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            terminators = [
                self.tokenizer.eos_token_id]

        generate_ids = self.model.generate(
            input_ids,
            top_p=0.9,
            temperature=0.5,
            max_length=2048,
            eos_token_id=terminators,
            min_length=input_len + 4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=7,
            do_sample=True,
        )

        outputs = self.tokenizer.batch_decode(
            generate_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        pred_sequence = outputs[0][len(text):]
        # pred_sequence = outputs[0]
        return pred_sequence

    def get_input_key(self):
        if self.args.input_key != '':
            return self.args.input_key

        # input_key = 'question_ar' if self.lang == AR else 'question'
        return self.input_keys[self.task]

    def process_multi_column_input_data(self, data, task=QUANT):
        ## Specific processing for quant data as it has multiple columns:
        for line in data:
            if line['choices'] not in line['direction']:
                line[self.input_keys[task]] = "\n".join([line['direction'], line['question'], line['choices']])
            else:
                ## Sometimes the choices are already present in direction, do not add choices in that case:
                line[self.input_keys[task]] = "\n".join([line['direction'], line['question']])

        return data

    def generate_response_file(self):
        if os.path.exists(self.output_file):
            print(
                f"For the generation of {self.task} the generation file already exists: {self.output_file}.\n Not generating again.")
            return
        input_key = self.get_input_key()

        data = read_jsonl(self.input_file)

        if self.task == QUANT:
            ## Specific processing for quant data as it has multiple columns:
            data = self.process_multi_column_input_data(data, self.task)

        # assert input_key in data[0].keys(), f"{input_key} does not exist in data: \n {data[0]}"

        prompt = self.get_prompt()
        response_data = []

        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        for d in tqdm(data):
            if d.get(input_key, None) is None:
                continue
            text = d[input_key]
            if self.args.use_chat_template:
                print("Using model's chat template formatting.")
                message = [{"role": "user", "content": text}]
                input_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            else:
                input_text = prompt.format_map({"Question": text})
            response = self.gen_text(input_text)

            d['question'] = text
            d['response'] = response
            d['input_text'] = input_text

            response_data.append(d)

        assert len(
            response_data) > 0, f"no output generated for file {self.input_file}\ncheck if {input_key} is present in the file."
        write_jsonl(response_data, self.output_file)
        print(f"Output responses written at {self.output_file}.")

    def generate_xresponse_file(self):
        if os.path.exists(self.output_file):
            print(
                f"For the generation of {self.task} the generation file already exists: {self.output_file}.\n Not generating again.")
            return
        input_key = self.get_input_key()

        data = read_jsonl(self.input_file)

        # assert input_key in data[0].keys(), f"{input_key} does not exist in data: \n {data[0]}"

        # prompt = self.get_prompt()
        response_data = []

        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        for d in tqdm(data):
            if d.get(input_key, None) is None:
                continue
            text = d[input_key]
            self.lang = d["prompt_lang"]
            prompt = self.get_prompt()
            input_text = prompt.format_map({"Question": text})
            response = self.gen_text(input_text)

            d['question'] = text
            d['response'] = response
            d['input_text'] = input_text

            response_data.append(d)

        assert len(
            response_data) > 0, f"no output generated for file {self.input_file}\ncheck if {input_key} is present in the file."
        write_jsonl(response_data, self.output_file)
        print(f"Output responses written at {self.output_file}.")

    def generate_gpt4_response_file(self):
        if os.path.exists(self.output_file):
            print(
                f"For the generation of {self.task} the generation file already exists: {self.output_file}.\n Not generating again.")
            return
        input_key = self.get_input_key()

        data = read_jsonl(self.input_file)

        # assert input_key in data[0].keys(), f"{input_key} does not exist in data: \n {data[0]}"

        response_data = []

        # self.tokenizer = self.load_tokenizer()
        # self.model = self.load_model()
        for d in tqdm(data):
            if d.get(input_key, None) is None:
                continue
            text = d[input_key]
            # prompt = self.get_prompt()
            gpt4_prompt = self.prompt["prompt_template"].format(instruction=text)
            response = gpt4_gen(self.prompt["system_prompt"], gpt4_prompt, 2048)

            d['question'] = text
            d['response'] = response
            d['input_text'] = gpt4_prompt

            response_data.append(d)

        assert len(
            response_data) > 0, f"no output generated for file {self.input_file}\ncheck if {input_key} is present in the file."
        write_jsonl(response_data, self.output_file)
        print(f"Output responses written at {self.output_file}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--task", type=str, choices=TASKS)
    parser.add_argument("--lang", type=str, choices=[AR, EN], default="en")
    parser.add_argument("--input_file", type=str, default='')
    parser.add_argument("--output_file", type=str, default='')
    parser.add_argument("--input_key", type=str, default='')
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--gpt_gen", action="store_true")
    parser.add_argument("--seq2seq", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_op = ModelOutput(args)

    if args.cross:
        model_op.generate_xresponse_file()
    elif args.gpt_gen:
        model_op.generate_gpt4_response_file()
    else:
        model_op.generate_response_file()
