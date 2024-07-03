import os
import glob
from random import shuffle

import pickle

ARA_DATA_DIR = "/nfs_users/users/onkar.pandit/code/lm-harness/datasets"

en_txt_files_path = "PaulGrahamEssays"
ar_txt_files_path = "ArabicText"

en_haystack_pkl = "en_haystack_context.pkl"
ar_haystack_pkl = "ar_haystack_context.pkl"

min_context_len = 500
max_context_len = 20100
context_len_step = 500
repeat_ques = 20 #

def write_pickle(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(file_path):
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    return data

pkl_path = os.path.join(ARA_DATA_DIR,en_haystack_pkl)

class GenHaystack:
    def __init__(self):
        self.haystack_txt_files_path = os.path.join(ARA_DATA_DIR, en_txt_files_path)
        self.haystack_pkl_path = os.path.join(ARA_DATA_DIR, en_haystack_pkl)

        self.needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        self.retrieval_question = "What is the best thing to do in San Francisco?"

        self.tokenizer = None
        self.txt_file_list = glob.glob(f"{self.haystack_txt_files_path}/*.txt")

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
                print(f"Actual context length: {cont_len}")
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



class GenHaystackAr(GenHaystack):
    def __init__(self):
        self.haystack_txt_files_path = os.path.join(ARA_DATA_DIR, ar_txt_files_path)
        self.haystack_pkl_path = os.path.join(ARA_DATA_DIR, ar_haystack_pkl)

        self.needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        self.retrieval_question = "What is the best thing to do in San Francisco?"

        self.tokenizer = None
        self.txt_file_list = glob.glob(f"{self.haystack_txt_files_path}/*.txt")



if __name__ == '__main__':
    gh = GenHaystack()
    gh.generate_haystack()

    gh = GenHaystackAr()
    gh.generate_haystack()