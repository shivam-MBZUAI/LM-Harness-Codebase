import torch
import os,sys
import transformers
from typing import Optional
from lm_eval.base import BaseLM

# iiai specific
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM,AutoTokenizer
from transformers import PreTrainedTokenizerFast

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict, PeftConfig
from lm_eval.models.btlm import register_btlm

class HFLM(BaseLM):
    def __init__(
        self,
        pretrained: str,
        device="cuda",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        device_list = set(
            ["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        )
        if device and device in device_list:
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        if revision != 'main':
            revision = revision + ("/" + subfolder if subfolder is not None else "")
            pretrained = pretrained + "/" +  revision

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        except:
            print(f"tokenizer.json not found at {pretrained}.\n Trying to load tokenizer with multilingual_v2.json.")
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=pretrained+'/multilingual_v2.json',
                eos_token="<|endoftext|>", pad_token="<|endoftext|>")
            # self.tokenizer = AutoTokenizer.from_pretrained("/nfs_shared/NLP_DATA/LM_DATA/models/llama_ablations_cbs/tokenizers/extend_100p/llama_tokenizer")
        print("tokenizer loaded")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                device_map="auto",trust_remote_code=True)
        except:
            # code to load LORA based models
            config = PeftConfig.from_pretrained(pretrained)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto",
                                                         trust_remote_code=True)
            model.resize_token_embeddings(len(self.tokenizer))
            self.model = get_peft_model(model, config)
        self.model.eval()


        device_maps = self.model.hf_device_map
        print("printing device maps:")
        print(device_maps)
        print(" ")


        self.vocab_size = self.tokenizer.vocab_size

        # setup for automatic batch size detection
        if batch_size == "auto":
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 2048

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps.to(self.device)).logits

    def _model_generate(self, context, max_length, eos_token_id):
        input_len = len(context)
        # generation_kwargs = {"do_sample": True, "max_length": 2048 - input_len,
        #                      "top_p": 0.95, "temperature": 0.35,
        #                      "repetition_penalty":1.2, "early_stopping":True}
        generation_kwargs = {"do_sample": True, "max_new_tokens": 2048,
                             "top_p": 0.95, "temperature": 0.35,
                             "repetition_penalty":1.2, "early_stopping":True}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        else:
            generation_kwargs['eos_token_id'] = self.eot_token_id
            generation_kwargs['pad_token_id'] = self.eot_token_id

        return self.model.generate(context, **generation_kwargs)

