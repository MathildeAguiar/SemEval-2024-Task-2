import os
import datasets

from datasets import load_dataset, concatenate_datasets


INSTRUCTION = "From the following statement and premise, would you say there is a contradiction between the statement and the premise ? If so, just answer by saying 'contradiction'.\n"
PREFIX_STATEMENT = "Statement: "
PREFIX_PREMISE = "Premise: "
PREFIX_ANSWER = "Answer: "
SEPARATOR_FS = "###\n"
PREFIX_EXPLANATION = "Explanation: "
PREFIX_WRONG_EXPLANATION = "Wrong explanation: "


def build_zs_instances():  # split
    """
        Build all the test templates from instances of the test set in a zero-shot setting, according to our defined template model.
        ADD THE TEMPLATE SCHEME

        returns: A set of built template (array of strings)
    """

    all_templates = []

    ds_train = load_dataset("bigbio/sem_eval_2024_task_2", split='train')
    ds_valid = load_dataset("bigbio/sem_eval_2024_task_2", split='validation')
    ds = concatenate_datasets([ds_train, ds_valid])
    #print("AAA", ds)
    # Shuffle the dataset (train+dev) to not have pattern or use always the same examples 
    nb_examples = round(ds.num_rows/3)
    ds = ds.shuffle().select(range(nb_examples))
    #print("BBB", ds)
    #print("CCC", nb_examples)

    # We take only 1/3 of the dev+train examples so we have the same number of examples for zero-shot, 1-shot and 2-shot
    for inst in ds:
        #print("DDD", inst)
        template = INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['premise'] + '\n'  # TODO test with the new vers of the ds 
        print("TEMP", template)
        all_templates.append(template)

    return all_templates


def build_1shot_instances():
    """
        Build all the templates from the test instances in a 1-shot setting, according to our defined template model.
        Corrected examples are taken from the shuffled train and dev sets.

        ADD THE TEMPLATE SCHEME 

        returns: Built template (string)
    """
    all_templates = []

    ds_train = load_dataset("bigbio/sem_eval_2024_task_2", split='train')
    ds_valid = load_dataset("bigbio/sem_eval_2024_task_2", split='validation')
    ds = concatenate_datasets([ds_train, ds_valid])
    # Shuffle the dataset (train+dev) to not have pattern or use always the same examples 
    nb_examples = round(ds.num_rows/3)
    ds = ds.shuffle()
    # Instances for inference 
    ds_inf = ds.select(range(nb_examples))
    # Instances for the 1-shot examples
    ds_1S = ds.select(range(nb_examples, nb_examples*2))
    # We take only 1/3 of the dev+train examples so we have the same number of examples for zero-shot, 1-shot and 2-shot
    # TODO on répète les mêmes instances et on doit récup les 2 autres slices 
    cnt = 0
    for inst in ds_inf:
            #print("DDD", inst)
            template = INSTRUCTION + PREFIX_STATEMENT + ds_1S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_1S[cnt-1]['type'] + '\n' + PREFIX_ANSWER + ds_1S[cnt-1]['label'] + '\n' + SEPARATOR_FS + INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['type'] 
            # TODO test with the new vers of the ds 
            print("TEMP", template)
            cnt+=1
            all_templates.append(template)

    return all_templates


def build_2shot_instances():
    """
        Build all the templates from the test instances in a 2-shot setting, according to our defined template model.
        Corrected examples are taken from the shuffled train and dev sets.

        ADD THE TEMPLATE SCHEME 

        returns: Built template (string)
    """

    all_templates = []

    ds_train = load_dataset("bigbio/sem_eval_2024_task_2", split='train')
    ds_valid = load_dataset("bigbio/sem_eval_2024_task_2", split='validation')
    ds = concatenate_datasets([ds_train, ds_valid])    
    nb_examples = round(ds.num_rows/3)
    ds = ds.shuffle().select(range(nb_examples))
    # We take only 1/3 of the dev+train examples so we have the same number of examples for zero-shot, 1-shot and 2-shot
    # TODO on répète les mêmes instances et on doit récup les 2 autres slices 
    for inst in ds:
            #print("DDD", inst)
            template = INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['type'] + '\n' + PREFIX_ANSWER + inst['label'] + '\n' + SEPARATOR_FS + '\n' +INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['type'] + '\n' + PREFIX_ANSWER + inst['label'] + '\n' +SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['type'] 
            # TODO test with the new vers of the ds 
            print("TEMP", template)
            all_templates.append(template)

    return all_templates


build_1shot_instances()