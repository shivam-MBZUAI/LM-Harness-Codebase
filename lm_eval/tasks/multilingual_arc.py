"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc
"""
from lm_eval.base import MultipleChoiceTask
from datasets import load_dataset
import os
from lm_eval.utils import ARA_DATA_DIR

_CITATION = """
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
"""

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hp,hq,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')
# Added:
#    "hp: IndicEval-ARC-Easy-hi", "hq: IndicEval-ARC-Challenge-hi"


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {arc_vi: Task, arc_bn: Task}
    """
    return {f"arc_{lang}": create_task(lang) for lang in LANGS}


def create_task(lang):

    class ATest(MultilingualARC):
        def __init__(self):
            super().__init__(lang)

    return ATest


class MultilingualARC(MultipleChoiceTask):

    def __init__(self, lang, **kwargs):
        self.VERSION = 0
        self.lang = lang
        self.DATASET_NAME = f"arc_{lang}"
        self.DATASET_PATH = 'multilingual_datasets/m_arc'
        self.NUM_FEW_SHOT = 25
        super().__init__(**kwargs)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": os.path.join(ARA_DATA_DIR,self.DATASET_PATH, f'{self.lang}_test.json'),
                                                        "train": os.path.join(ARA_DATA_DIR,self.DATASET_PATH, f'{self.lang}_train.json'),
                                                        "validation": os.path.join(ARA_DATA_DIR,self.DATASET_PATH, f'{self.lang}_validation.json')})

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # NOTE:
        options = ['option_a','option_b','option_c','option_d']
        doc["choices"] = [doc[choice] for choice in options]
        out_doc = {
            "id": doc["id"],
            "query": "Question: " + doc["instruction"] + "\nAnswer:",
            "choices": doc["choices"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answer"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
