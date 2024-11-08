"""
This is manually digitised QA dataset. These are MCQs from Arabic Language Course at https://www.madinaharabic.com/quiz/
TODO: Add more details
"""

from lm_eval.base import rf, Task
from datasets import load_dataset
import numpy as np
import os.path as osp
from lm_eval.metrics import mean, matthews_corrcoef, f1_score_multiclass, f1_score, yesno
from lm_eval.utils import general_detokenize, camel_clean, PROMPT_DICT, ARA_DATA_DIR

_CITATION = """
"""


class GrammarQA(Task):
    VERSION = 0 # contains 413 MCQ questions with multiple options and 1 correct choice.

    def __init__(self, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'grammar_ar.jsonl')})


    def _process_doc(self, doc):
        que = camel_clean(doc["question"])
        out_doc = {
            "id": doc["id"],
            "question": que,
            "query": "Question: " + que + "\nAnswer:",
            "choices": [camel_clean(t) for t in doc["choices"]["text"]],
            "gold": doc["choices"]["label"].index(doc["answerKey"]),
        }
        return out_doc

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self._process_doc, self.dataset["train"]))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]
        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]
        pred = np.argmax(results)

        acc = 1.0 if pred == gold else 0.0

        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {"acc": acc, "acc_norm": acc_norm, "macro_f1": (pred, gold)}

    def aggregation(self):
        return {"acc": mean, "acc_norm": mean, "macro_f1": f1_score_multiclass}

    def higher_is_better(self):
        return {"acc": True, "acc_norm": True, "macro_f1": True}


class GrammarRC(Task):
    VERSION = 0

    def __init__(self, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'madinah_rc.jsonl')})


    def _process_doc(self, doc):
        que = camel_clean(doc["question"])
        context = camel_clean(doc["context"])
        out_doc = {
            "id": doc["id"],
            "question": que,
            "context": context,
            "query": "Question: " + que + "\nPassage: " + context + "\nAnswer:",
            "choices": [camel_clean(t) for t in doc["choices"]["text"]],
            "gold": doc["choices"]["label"].index(doc["answerKey"]),
        }
        return out_doc

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self._process_doc, self.dataset["train"]))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]
        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]
        pred = np.argmax(results)

        acc = 1.0 if pred == gold else 0.0

        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {"acc": acc, "acc_norm": acc_norm, "macro_f1": (pred, gold)}

    def aggregation(self):
        return {"acc": mean, "acc_norm": mean, "macro_f1": f1_score_multiclass}

    def higher_is_better(self):
        return {"acc": True, "acc_norm": True, "macro_f1": True}