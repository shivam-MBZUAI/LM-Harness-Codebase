from . import gpt2
from . import gpt3
from . import huggingface
from . import textsynth
from . import dummy

# IIAI Models
from . import cerebras, iiai_cerebras
# from . import vllm

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "hf-causal": gpt2.HFLM,
    "hf-causal-experimental": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
    # IIAI models
    "cerebras": cerebras.HFLM,
    "iiai-cerebras": iiai_cerebras.HFLM,
    # "vllm": vllm.VLLM
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
