from typing import Dict, List, Optional
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer, HfArgumentParser
import logging
from dataclasses import dataclass, field
from .preprocess_llm import (
    build_1shot_instances, build_1shot_instances_CCOT, build_1shot_instances_COT, build_2shot_instances, 
    build_2shot_instances_CCOT, build_2shot_instances_COT, build_zs_instances
    )

from .post_process_llm import parse_label, format_results


logger = logging.getLogger(__name__)

#################################################################################


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/template we are going to prompt from.
    """

    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    template_name: str = field(
        default=None, 
        metadata={"help": "Name of the template to pass to the model. Templates available: ZS, 1S, 2S, 1S_COT, 2S_COT, 1S_CCOT, 2S_CCOT."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


###################################################################################


def main():
    ### Parse the arguments ###
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args_into_dataclasses()
    # Setup logging TODO see if we need more logs 
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    ### Build the prompts ###

    # Build the prompts in fct of the argument passed 
    if model_args.template_name == 'ZS':
        templates, all_ids = build_zs_instances()
    elif model_args.template_name == '1S':
        templates, all_ids = build_1shot_instances()
    elif model_args.template_name == '2S':
        templates, all_ids = build_2shot_instances()
    elif model_args.template_name == '1S_COT':
        templates, all_ids = build_1shot_instances_COT()
    elif model_args.template_name == '2S_COT':
        templates, all_ids = build_2shot_instances_COT()
    elif model_args.template_name == '1S_CCOT':
        templates, all_ids = build_1shot_instances_CCOT()
    elif model_args.template_name == '2S_CCOT':
        templates, all_ids = build_2shot_instances_CCOT()
    
    #### Instanciate the desired model and tokenizer ###
    model_name_or_path = model_args.model_name_or_path
    # TODO DELETE "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", use_fast=False, token=model_args.token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        #revision=revision,
        torch_dtype=torch.float16,
        device_map="auto",
        token=model_args.token
        # load_in_8bit=True,
        # low_cpu_mem_usage=True,
    )

    streamer = TextStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    def chat(
        query: str,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: float = 0,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1024,
        **kwargs,
    ):
        if history is None:
            history = []

        history.append({"role": "user", "content": query})

        input_ids = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
        input_length = input_ids.shape[1]

        generated_outputs = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                temperature=temperature,
                do_sample=temperature > 0.0,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            ),
            streamer=streamer,
            return_dict_in_generate=True,
        )

        generated_tokens = generated_outputs.sequences[0, input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        history.append({"role": "assistant", "content": generated_text})

        return generated_text, history

    output_answers = []
    for prompt in templates:
        response, history = chat(prompt, history=None)
        output_answers.append(response)

    #### Parse the outputs to get the predicted label ####
    # Parsing
    preds = parse_label(output_answers)
    # Formating into the challenge's format 
    format_results(all_ids, preds)



if __name__ == "__main__":
    main()
