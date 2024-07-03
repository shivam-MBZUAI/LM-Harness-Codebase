"""
TODO

"""
import re
from lm_eval.base import PerplexityTask
import pandas as pd

from datasets import load_dataset
import os.path as osp
from lm_eval.tasks.quality import QUALITY

_CITATION = """
"""

DATA_DIR = r"/l/users/haonan.li/iiai_llm/llm-eval/datasets/ardc"

class ARDC(PerplexityTask):
    VERSION = 0
    DATASET_PATH = ""
    DATASET_NAME = ""

    def __init__(self, data_dir=None, file_name=None, cache_dir=None, download_mode=None):
        self.download(data_dir, file_name, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, file_name=None, cache_dir=None, download_mode=None):
        dataset = pd.read_pickle(f"{DATA_DIR}/test_data.pkl")
        self.dataset = dataset

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            for doc in self.dataset["text"]:
                yield doc

    def should_decontaminate(self):
        return True



class ARDCLongContext(QUALITY):
    VERSION = 0
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'ardc_long.jsonl')})

    def _process_doc(self, doc):
        # TODO: add context to query
        que = doc["question"]
        out_doc = {
            "id": doc["id"],
            "context": "",
            "question": que,
            "query": "Document: \n" + doc["question"] + "\n  What is the topic of this document? \nAnswer:",
            "choices": doc["choices"]["text"],
            "gold": doc["choices"]["label"].index(doc["answerKey"]),
        }
        return out_doc
