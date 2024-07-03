"""
QUALITY:
Homepage: https://huggingface.co/datasets/L4NLP/LEval/viewer/quality?row=0
"""

from datasets import load_dataset
import os.path as osp
from lm_eval.tasks.quality import QUALITY


_CITATION = """
"""


class TPO(QUALITY):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'tpo_test.jsonl')})


