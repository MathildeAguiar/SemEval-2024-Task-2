import os 
import datasets
import evaluate
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)



tokenizer = AutoTokenizer.from_pretrained()

def preprocess_function(examples):
    # Tokenize the texts, 2 different options if its a comparision or a single CTR
    #TODO

    return tokenizer(
        examples["primary_premise"],
        examples["secondary_premise"],
        examples['statement'],
        padding=padding,
        max_length=data_args.max_seq_length,
        truncation=True,
        )