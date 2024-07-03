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

Homepage: https://github.com/hendrycks/test
"""
import os.path as osp
from datasets import load_dataset
from lm_eval.utils import general_detokenize, ARA_DATA_DIR
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
    task_dict = {f"mmlu_ar-{sub}": create_task(sub) for sub in SUBJECTS}
    return task_dict


def create_task(subject):
    class MMLU_AR(GeneralHendrycksTest):
        def __init__(self):
            super().__init__(subject)

    return MMLU_AR


class GeneralHendrycksTest(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "translated_dataset/MMLU"
    DATASET_NAME = None

    def __init__(self, subject, cache_dir=None, download_mode=None):
        self.DATASET_NAME = subject
        self.download(ARA_DATA_DIR, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={
            "test": osp.join(data_dir, self.DATASET_PATH, f"test/{self.DATASET_NAME}_test.csv"),
            "validation": osp.join(data_dir, self.DATASET_PATH, f"val/{self.DATASET_NAME}_val.csv"), })

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
            prompt = "Question: " + doc["0"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, [doc["1"], doc["2"], doc["3"], doc["4"]])]
            )
            prompt += "Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": [doc["1"], doc["2"], doc["3"], doc["4"]],  # doc["choices"],
            "gold": keys.index(doc["5"])
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["validation"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


# -------------------------------------------------------- #
# Human translated tasks
SUBJECTS_HU = [
    'logical_fallacies',
    'high_school_government_and_politics',
    'college_mathematics',
    'us_foreign_policy',
    'high_school_microeconomics',
    'anatomy',
    'public_relations',
    'high_school_us_history',
    'management',
    'business_ethics',
    'machine_learning',
    'elementary_mathematics',
    'abstract_algebra',
    'college_physics',
    'moral_disputes',
    'virology',
    'world_religions',
    'college_computer_science',
    'global_facts',
    'econometrics',
    'sociology',
    'high_school_geography',
    'human_aging',
    'international_law',
    'human_sexuality',
    'computer_security',
    'college_biology',
    'medical_genetics',
    'clinical_knowledge',
    'high_school_mathematics',
    'high_school_computer_science',
    'high_school_physics',
    'philosophy',
    'jurisprudence',
    'prehistory',
    'electrical_engineering',
    'moral_scenarios',
    'college_chemistry',
    'security_studies',
    'conceptual_physics',
    'high_school_world_history',
    'college_medicine',
    'nutrition',
    'formal_logic',
    'astronomy',
    'marketing',
    'high_school_psychology',
    'professional_medicine',
    'miscellaneous',
    'professional_law',
    'high_school_macroeconomics',
    'professional_psychology',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_statistics',
    'professional_accounting',
    'high_school_european_history',
]


def create_all_hu_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {MMLU-ar-abstract_algebra: Task}
    """
    task_dict = {f"mmlu_hu_ar-{sub}": create_hu_task(sub) for sub in SUBJECTS_HU}
    return task_dict


def create_hu_task(subject):
    class MMLU_HU_AR(GeneralHendrycksTestHuman):
        def __init__(self):
            super().__init__(subject)

    return MMLU_HU_AR


class GeneralHendrycksTestHuman(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "translated_dataset/MMLU_human"
    DATASET_NAME = None

    def __init__(self, subject, cache_dir=None, download_mode=None):
        self.DATASET_NAME = subject
        self.download(ARA_DATA_DIR, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={
            "test": osp.join(data_dir, self.DATASET_PATH, f"test1/{self.DATASET_NAME}.jsonl"),
            })

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

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
            prompt = "Question: " + doc["question"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt += "Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
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
            self._fewshot_docs = list(map(self._process_doc, self.dataset["validation"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
