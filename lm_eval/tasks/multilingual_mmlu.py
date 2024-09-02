"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The Hendryck's Test is a benchmark that measured a text model’s multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model’s academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://github.com/hendrycks/test
"""
from lm_eval.base import MultipleChoiceTask

from datasets import load_dataset
import os
from lm_eval.utils import ARA_DATA_DIR

_CITATION = """
@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
"""
LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hm,hn,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')
# Added """hm""" for Meta Llama-3.1 mmlu hindi evaluation set.
# Source: "https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals/viewer/Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_hi__details"
# Paper: "https://scontent.ffjr1-6.fna.fbcdn.net/v/t39.2365-6/452387774_1036916434819166_4173978747091533306_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=DTS7hDTcxZoQ7kNvgGKLgOY&_nc_ht=scontent.ffjr1-6.fna&oh=00_AYAVW94oy05nlimqg6bU2Uv0gWib6izVJ_BVIMkb6k6kkw&oe=66AE9C4D"
# MMLU Paper: "https://arxiv.org/pdf/2009.03300"

# Added:
#    "hn: IndicEval-mmlu-hi"


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra: Task, hendrycksTest-anatomy: Task}
    """
    return {f"mmlu_{lang}": create_task(lang) for lang in LANGS}


def create_task(lang):
    class HendrycksTest(GeneralHendrycksTest):
        def __init__(self):
            super().__init__(lang)

    return HendrycksTest


class GeneralHendrycksTest(MultipleChoiceTask):
    VERSION = 0
    NUM_FEW_SHOT = 25
    DATASET_PATH = "multilingual_datasets/m_mmlu"
    DATASET_NAME = None

    def __init__(self, lang):
        self.DATASET_PATH = "multilingual_datasets/m_mmlu"
        self.DATASET_NAME = f'mmlu_{lang}'
        self.lang = lang
        super().__init__()

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": os.path.join(ARA_DATA_DIR,self.DATASET_PATH, f'{self.lang}_test.json'),
                                                        # "train": os.path.join(ARA_DATA_DIR,self.DATASET_PATH, f'{self.lang}_train.json'),
                                                        "validation": os.path.join(ARA_DATA_DIR,self.DATASET_PATH, f'{self.lang}_dev.json')})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Question: <prompt>
            Choices:
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """
            prompt = "Question: " + doc["instruction"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {doc[choice]}\n" for key, choice in zip(keys, options)]
            )
            prompt += "Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        options = ['option_a','option_b','option_c','option_d']
        doc["choices"] = [doc[choice] for choice in options]
        return {
            "query": format_example(doc, keys),
            "choices": doc["choices"],
            "gold": keys.index(doc["answer"])
            if isinstance(doc["answer"], str)
            else doc["answer"],
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
