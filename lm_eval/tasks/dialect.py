"""
This is manually Created QA dataset.
TODO: Add more details
"""

from datasets import load_dataset
import os.path as osp

from lm_eval.tasks.digitised import DigitisedQA_AR

_CITATION = """
"""


class GULF_DIALECT(DigitisedQA_AR):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'gulf_dialect.jsonl')})
