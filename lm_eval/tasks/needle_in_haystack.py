
"""
The evaluation works in the following way.  Place a random fact or statement (the 'needle') in the middle of a long
context window (the 'haystack’). Then the model is asked to retrieve this statement. The predicted answer is compared
to the gold fact which we know. This process can be iterated over various document depths (where the needle is placed)
and context lengths to measure performance.

Homepage: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
"""

import os

from lm_eval.base import rf, Task
from lm_eval.metrics import evaluate_response,mean
from lm_eval.utils import general_detokenize, camel_clean, ARA_DATA_DIR,read_json,write_pickle,read_pickle

import glob
from random import shuffle


_CITATION = """
"""

en_txt_files_path = "PaulGrahamEssays"
ar_txt_files_path = "ArabicText"

# en_haystack_json_file = "en_haystack_context.json"
# ar_haystack_json_file = "ar_haystack_context.json"

en_haystack_pkl = "en_haystack_context.pkl"
ar_haystack_pkl = "ar_haystack_context.pkl"

fert_factor = 1.3

min_context_len = 500
max_context_len = 8000
context_len_step = 500

depth_steps = 10
repeat_ques = 10 # averaging over scores from the model by prompting multiple times to get better understanding

class NeedleInHaystack(Task):
    VERSION = 0

    def __init__(self):
        self.haystack_txt_files_path = os.path.join(ARA_DATA_DIR, en_txt_files_path)
        self.haystack_pkl_path = os.path.join(ARA_DATA_DIR,en_haystack_pkl)

        self.needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        self.retrieval_question = "What is the best thing to do in San Francisco?"

        self.tokenizer = None
        self.txt_file_list = glob.glob(f"{self.haystack_txt_files_path}/*.txt")

        self._min_context_len = min_context_len
        self._max_context_len = max_context_len
        self._context_len_step = context_len_step

    @property
    def min_context_len(self):
        return self._min_context_len

    @property
    def max_context_len(self):
        return self._max_context_len

    @property
    def context_len_step(self):
        return self._context_len_step

    @min_context_len.setter
    def min_context_len(self, value):
        if value is None:
            self._min_context_len = min_context_len
        else:
            if value < 0:
                raise ValueError("min_context_len must be a positive integer.")
            self._min_context_len = value

    @max_context_len.setter
    def max_context_len(self, value):
        if value is None:
            self._max_context_len = max_context_len
        else:
            if value < self._min_context_len:
                raise ValueError("max_context_len must be greater than min_context_len.")
            self._max_context_len = value

    @context_len_step.setter
    def context_len_step(self, value):
        if value is None:
            self._context_len_step = context_len_step
        else:
            if value != 500:
                raise ValueError("Right now cotext hop should be 500. If you want to change it to some other steps, you will have to generate new haystack data.")
            self._context_len_step = value

    def _is_context_of_len(self, context, context_len):
        if not self.tokenizer:
            self._load_tokenizer()
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids
        if len(input_ids[0]) < context_len:
            return None,None
        else:
            input_ids = self.tokenizer(context, max_length=context_len, truncation=True, return_tensors="pt").input_ids
            cont_len = len(input_ids[0])
            return self.tokenizer.decode(input_ids[0]),cont_len

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("core42/jais-13b-chat")

    def get_haystack_of_spec_length(self, haystack_len):
        shuffle(self.txt_file_list)
        context = ""
        for file in self.txt_file_list:
            # print(file)
            with open(file, 'r') as f:
                context += f.read()
            exact_len_context,cont_len = self._is_context_of_len(context, haystack_len)
            if exact_len_context:
                print(f"Wanted context len: {haystack_len} and Actual context length: {cont_len}")
                return exact_len_context
    def generate_haystack(self):
        context_len_haystack_dict = {}
        for curr_context_len in range(min_context_len, max_context_len, context_len_step):
            print(f"Expected context length: {curr_context_len}")
            for cc in range(repeat_ques):
                print(cc)
                _context_list = context_len_haystack_dict.get(curr_context_len, [])
                current_context_length_context = self.get_haystack_of_spec_length(curr_context_len)
                _context_list.append(current_context_length_context)
                context_len_haystack_dict[curr_context_len] = _context_list
            print(5 * "---")
            print("\n\n\n")
        write_pickle(context_len_haystack_dict, self.haystack_pkl_path)
        return context_len_haystack_dict

    def read_haystack(self):
        print(f"Reading haystack file: {self.haystack_pkl_path}")
        context_len_haystack_dict = read_pickle(self.haystack_pkl_path)
        # context_len_haystack_dict = {int(k):context_len_haystack_dict[k] for k in context_len_haystack_dict.keys()}
        return context_len_haystack_dict

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        doc = []
        self.context_len_haystack_dict = self.read_haystack()

        for haystack_len in range(self.max_context_len,self.min_context_len-self.context_len_step,-self.context_len_step):
            print(f"Haystack length: {haystack_len}\n")
            for depth in range(0,101,depth_steps):
                for repeat in range(repeat_ques):
                    num_words = int(haystack_len/fert_factor)

                    needle_words = self.needle.split()
                    num_needle_words = int(len(needle_words)/fert_factor)

                    context = self.context_len_haystack_dict[haystack_len][repeat]
                    # context = self.get_haystack_of_spec_length(haystack_len)
                    context_list = context.split()[0:(num_words-num_needle_words)]

                    # input_to_model = context_list[:needle_location] + self.needle.split() + context_list[needle_location+1:]
                    if depth ==0:
                        context_list = needle_words + context_list
                    elif depth == 100:
                        context_list = context_list + needle_words
                    else:
                        needle_location = int(len(context_list) * depth / 100) - 1
                        context_list[needle_location:needle_location] = needle_words

                    # input_to_model = context_list[0:num_words]

                    input_to_model = " ".join(context_list)
                    ddict = {'context_len':haystack_len,'depth':depth,"source":f"{input_to_model}\n{self.retrieval_question}"}
                    doc.append(ddict)
        return doc

    def doc_to_text(self, doc):
        return doc['source']

    def doc_to_target(self, doc):
        return self.needle

    def construct_requests(self, doc, ctx):
        try:
            continuation = rf.greedy_until(ctx, {"until": ["<|endoftext|>"]})
        except Exception as e:
            continuation = ""
            print("No output can be found from the llm.")
            print(e)
        return continuation

    def process_results(self, doc, results, save_all=False):
        completion = results[0].strip()
        score = evaluate_response(self.retrieval_question,self.needle,completion)

        result = {
            "score": score,
        }

        if save_all:
            result["example"] = {
                'context_len':doc['context_len'],
                'depth':doc['depth'],
                # "input": doc["source"],
                "ref": self.needle,
                "pred": completion,
                "score":score
            }

        return result

    def aggregation(self):
        return {
            "score":mean
        }

    def higher_is_better(self):
        return {
            "score": True
        }


class NeedleInHaystackAr(NeedleInHaystack):
    VERSION = 0

    def __init__(self):
        self.haystack_txt_files_path = os.path.join(ARA_DATA_DIR,ar_txt_files_path)
        self.haystack_pkl_path = os.path.join(ARA_DATA_DIR, ar_haystack_pkl)

        self.needle = "\nأفضل ما يمكنك فعله في أبو ظبي هو احتساء القهوة الذهبية عيار 22 قيراط والاسترخاء في قصر الإمارات.\n" #The best thing to do in Abu Dhabi is to drink 22k gold coffee and relax at Emirates Palace.
        self.retrieval_question = "ما هو أفضل شيء يمكنك القيام به في أبو ظبي؟" #What is the best thing to do in Abu Dhabi?

        self.tokenizer = None
        self.txt_file_list = glob.glob(f"{self.haystack_txt_files_path}/*.txt")
