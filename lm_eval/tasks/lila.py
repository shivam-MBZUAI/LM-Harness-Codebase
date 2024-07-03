"""
The dataset contains mathematics questions, output programs which are python code and actual answer to the question.
In their approach they obtained python code for the question and executed that code to get the final answer.
This answer is directly compared with the actual gold answer. Then F1 score is calculate to get the evaluation score.
Their dataset is composition from different 23 datasets and each question is marked on four different verticles as maths,
used language (complex or simple), knowledge (common sense required or not), format (Fill in the blanks, mcq).

Homepage: https://huggingface.co/datasets/allenai/lila
"""
_CITATION = """
@INPROCEEDINGS{Mishra2022Lila,
  author = {
    Swaroop Mishra 
      and Matthew Finlayson 
      and Pan Lu 
      and Leonard Tang 
      and Sean Welleck 
      and Chitta Baral 
      and Tanmay Rajpurohit 
      and Oyvind Tafjord 
      and Ashish Sabharwal 
      and Peter Clark 
      and Ashwin Kalyan},
  title = {Lila: A Unified Benchmark for Mathematical Reasoning},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2022}
}
"""

from lm_eval.base import rf, Task
from lm_eval.metrics import evaluate_response, mean
from datasets import load_dataset

import sys
from io import StringIO
import contextlib


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


sub_datasets = ['APPS_structured', 'GSM8k_structured', 'MATH_algebra_crowdsourced',
                'MATH_counting_and_probability_crowdsourced',
                'MATH_intermediate_algebra_crowdsourced', 'MCTaco_event_duration_structured',
                'MCTaco_event_ordering_structured',
                'MCTaco_event_typical_time_structured', 'MCTaco_frequency_structured', 'MCTaco_stationarity_structured',
                'NumGLUE_Type_1_crowdsourced', 'NumGLUE_Type_2_crowdsourced', 'NumGLUE_Type_3_crowdsourced',
                'NumGLUE_Type_4_crowdsourced', 'NumGLUE_Type_5_crowdsourced', 'NumGLUE_Type_6_crowdsourced',
                'NumGLUE_Type_7_crowdsourced', 'NumGLUE_Type_8_crowdsourced', 'Numersense_structured', 'addsub',
                'amps_algebra',
                'amps_calculus', 'amps_counting_and_stats', 'amps_geometry', 'amps_linear_algebra',
                'amps_number_theory', 'asdiv',
                'conala_structured', 'deepmind_mathematics_algebra', 'deepmind_mathematics_basicmath',
                'deepmind_mathematics_calculus', 'deepmind_mathematics_muldiv', 'deepmind_mathematics_numbertheory',
                'dolphin_t2_final', 'draw_structured', 'mathqa_gain', 'mathqa_general', 'mathqa_geometry',
                'mathqa_other',
                'mathqa_physics', 'mathqa_probability', 'mbpp_structured', 'multiarith', 'simuleq', 'singleop',
                'singleq',
                # 'iid', 'ood',
                'svamp_structured']


def create_all_tasks():
    task_dict = {f"lila-{sub}": create_task(sub) for sub in sub_datasets}
    return task_dict


def create_task(sub_dataset):
    class Lila(GeneralLila):
        def __init__(self):
            super().__init__(sub_dataset)

    return Lila


class GeneralLila(Task):
    VERSION = 0

    def __init__(self, sub_dataset):
        self.dataset = load_dataset("allenai/lila", sub_dataset)
        self._training_docs = None
        self._fewshot_docs = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return 'validation' in self.dataset.keys()

    def has_test_docs(self):
        return 'test' in self.dataset.keys()

    def training_docs(self):
        return self.dataset["validation"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        if len(self.dataset['test'])>250:
            return self.dataset["test"].select(range(250))
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"Write a python program to solve the following question. Make sure to write a print statement at the end to output the final answer.\n{doc['input']}\nAnswer\n"

    def doc_to_target(self, doc):
        return doc['output_program']

    def construct_requests(self, doc, ctx):
        try:
            continuation = rf.greedy_until(ctx, {"until": ["<|endoftext|>"]})
        except Exception as e:
            continuation = ""
            print("No output can be found from the llm.")
            print(e)
        return continuation

    def run_python_code(self, code):
        with stdoutIO() as s:
            try:
                exec(code)
            except Exception as exception:
                err_type = type(exception).__name__
                # print(exception)
                return "**##NO OUTPUT##**", err_type, exception
            return s.getvalue(), "NO ERROR", "pass"

    def process_results(self, doc, results, save_all=False):
        # import ipdb;
        # ipdb.set_trace()

        completion = results[0].strip()
        program_output, err_type, exception = self.run_python_code(completion)

        acc = 1.0 if program_output == doc["output_answer"] else 0.0
        result = {"acc": acc}

        if save_all:
            result["example"] = {
                'input': doc['input'],
                'output_program': doc['output_program'],
                "output_answer": doc["output_answer"],
                "pred_program": completion,
                "pred_answer": program_output,
                "err_type": err_type,
                # "actual_exception":exception,
                "acc": acc,
            }

        return result

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
