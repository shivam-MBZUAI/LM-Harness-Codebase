"""
A machine translated version of MMLU, translate from English to Arabic.

Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The Hendryck's Test is a benchmark that measured a text model’s multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model’s academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://huggingface.co/datasets/OALL/Arabic_MMLU
"""
import os.path as osp
from datasets import load_dataset
from lm_eval.utils import general_detokenize, ARA_DATA_DIR, camel_clean, PROMPT_DICT
from lm_eval.base import MultipleChoiceTask

_CITATION = """
@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
"""


# Remove some tasks to fit it into 12 hours
SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {MMLU_ar-abstract_algebra: Task}
    """
    task_dict = {f"oall_mmlu_ar-{sub}": create_task(sub) for sub in SUBJECTS}
    return task_dict


def create_task(subject):
    class OALL_MMLU_AR(OALL_MMLU_AR_Subject):
        def __init__(self):
            super().__init__(subject)

    return OALL_MMLU_AR


# TODO: Replace `NewTask` with the name of your Task.
class OALL_MMLU_AR_Subject(MultipleChoiceTask):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "OALL/Arabic_MMLU"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def __init__(self, subject, cache_dir=None, download_mode=None):
        self.DATASET_NAME = subject
        self.download(self.DATASET_PATH, cache_dir, download_mode)
        # self._training_docs = None
        # self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # self.dataset = {}
        # for subjct in SUBJECTS:
        #     self.dataset[subjct] = load_dataset(self.DATASET_PATH, subjct)
        self.dataset = load_dataset(self.DATASET_PATH, self.DATASET_NAME)
        # self.dataset = load_dataset("csv", data_files={"test":osp.join(data_dir, self.DATASET_PATH, f"test/{self.DATASET_NAME}_test.csv"),
        #                                                "validation":osp.join(data_dir, self.DATASET_PATH, f"val/{self.DATASET_NAME}_val.csv"),})

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return False

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["dev"])

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:

        """
        Question: <prompt>
        Choices:
        A. <choice1>
        B. <choice2>
        C. <choice3>
        D. <choice4>
        Answer:
        """
        que = camel_clean(doc["question"])
        choice_keys = ["A", "B", "C", "D"]
        choices = "".join(
                [f"{c}. {doc[c]}\n" for c in choice_keys]
            )
        out_doc = {
            # "id": doc["id"],
            "subject": doc["subject"],
            "subset": self.DATASET_NAME,
            "question": que,
            "query": f"Question: {que}\nChoices:{choices}\nAnswer:",  # The query prompt.
            "choices": [camel_clean(doc[c]) for c in choice_keys],  # The list of choices.
            "gold": choice_keys.index(doc["answer"]),  # The integer used to index into the correct element of `"choices"`.
        }
        return out_doc


    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        return rnd.sample(list(self._fewshot_docs), k)
