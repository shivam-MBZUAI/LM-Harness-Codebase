"""
SIQA
"""
from lm_eval.base import MultipleChoiceTask
import os.path as osp
from datasets import load_dataset
from lm_eval.utils import ARA_DATA_DIR


_CITATION = """
SIQA
"""


class SiQA_AR(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "translated_dataset/siqa"
    DATASET_NAME = None
    
    def __init__(self, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"validation":osp.join(data_dir, self.DATASET_PATH, f"validation.csv"),})

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["context"] + '\n' + doc['question'],
            "choices": [doc["answerA"], doc["answerB"], doc["answerC"]],
            "gold": int(doc["label"])-1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return "Question: " + doc["query"] + "\nAnswer: "

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["goal"]
