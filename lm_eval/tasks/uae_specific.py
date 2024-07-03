"""
This is manually digitsed QA dataset. Question papers from different universities have been converted into MCQ questions.
TODO: Add more details
"""

from datasets import load_dataset
import os.path as osp

from lm_eval.tasks.digitised import DigitisedQA_AR

_CITATION = """
"""


class UAE_Specific_AR(DigitisedQA_AR):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'UAE_ar.jsonl')})


class UAE_Specific_EN(DigitisedQA_AR):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'UAE_en.jsonl')})