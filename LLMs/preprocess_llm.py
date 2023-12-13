import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
# from build_dataset import add_premises_to_dataset


INSTRUCTION = "From the following statement and premise, would you say there is a contradiction between the statement and the premise ? If so, just answer by saying 'contradiction'.\n"
PREFIX_STATEMENT = "Statement: "
PREFIX_PREMISE = "Premise: "
PREFIX_ANSWER = "Answer: "
SEPARATOR_FS = "###" 
PREFIX_EXPLANATION = "Explanation: Considering the following evidences in the premise: '"
MIDDLE_EXPLANATION = "', we can conclude that there is a "
SUFFIX_EXPLANATION = " between the premise and the statement."
PREFIX_WRONG_EXPLANATION = "Wrong explanation: Considering the following evidences in the premise: '"
MIDDLE_WRONG_EXPLANATION = "', we can conclude that there is a "
SUFFIX_WONRG_EXPLANATION = " between the premise and the statement."
PATH_TO_LOCAL_DS = "/home/aguiar/Documents/SE_2024_Task_2/SemEval-2024-Task-2/training_data/sampled_data"
# PATH_TO_LOCAL_DS = "/mnt/beegfs/home/aguiar/SE2024/sampled_data"

### Utils ###

def concat_premise(example):
    concat_premise = ""
    for i in example['primary_premise']:
        concat_premise = concat_premise + str(i) + '\n'
    if example['type'] == 'Comparison':
        for j in example['secondary_premise']:
            concat_premise = concat_premise + str(j) + '\n'
    example['primary_premise'] = concat_premise
    return example

def concat_evidence(example):
    concat_evidence = ""
    if 'primary_evidences' in example.keys():  # TODO see why there is 1 instance that does not have evidences 
        for i in example['primary_evidences']:
            concat_evidence = concat_evidence + str(i) + ' '
        if example['type'] == 'Comparison':
            for j in example['secondary_evidences']:
                concat_evidence = concat_evidence + str(j) + ' '
        example['primary_evidences'] = concat_evidence
        return example 
    else:
        pass

def pick_wrong_evidence(example):
    # TODO améliorer 
    """
    concat_wrong_evidence = ""
    cnt1 = 0
    cnt2 = cnt1+1
    wrong_evid1 = example['primary_premise'][cnt1]
    wrong_evid2 = example['primary_premise'][cnt2]
    if wrong_evid1 not in example['primary_evidences']:
        if wrong_evid2 not in example['primary_evidences']:
           concat_wrong_evidence = wrong_evid1 + ' ' + wrong_evid2
           return concat_wrong_evidence
        else:
            cnt1+=1
    """
    concat_wrong_evidence = ""
    if 'primary_evidences' in example.keys():  # TODO see why there is 1 instance that does not have evidences 
        available_elements = [element for element in example['primary_premise'] if element not in example['primary_evidences']]
        if len(available_elements) < 2:  # NOTE: This case truly exists, what do we do ??
            concat_wrong_evidence = 'None'
        else:
            concat_wrong_evidence = available_elements[0] + available_elements[1]
        example['wrong_evidences'] = concat_wrong_evidence
        return example
    else:
        pass
            
        


#### For classic prompting ####

def build_zs_instances():
    """
        Build all the test templates from instances of the test set in a zero-shot setting, according to our defined template model.
        ADD THE TEMPLATE SCHEME

        returns: A set of built template (array of strings)
    """

    all_templates = []
    # Need the ids for formating the results 
    all_ids = []

    # og_ds = add_premises_to_dataset()
    # Load from local files
    og_ds = load_from_disk(PATH_TO_LOCAL_DS)
    ds_test = og_ds['test']
    # Format the premise 
    ds_test = ds_test.map(concat_premise)

    for inst in ds_test:
        template = INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['primary_premise'] + '\n' 
        all_templates.append(template)
        all_ids.append(inst['id'])  
    print("LEN", len(all_templates))
    print(all_templates[0])

    return all_templates, all_ids

# build_zs_instances()

def build_1shot_instances():
    """
        Build all the templates from the test instances in a 1-shot setting, according to our defined template model.
        Corrected examples are taken from the shuffled train and dev sets.

        ADD THE TEMPLATE SCHEME 

        returns: Built template (string)
    """
    all_templates = []
    all_ids = []

    # Load from local files
    og_ds = load_from_disk(PATH_TO_LOCAL_DS)
    ds_train = og_ds['train']
    ds_valid = og_ds['validation']
    ds_test = og_ds['test']
    # Format the premise 
    ds_train = ds_train.map(concat_premise)
    ds_valid = ds_valid.map(concat_premise)
    ds_test = ds_test.map(concat_premise)

    ds = concatenate_datasets([ds_train, ds_valid])
    # Shuffle the dataset (train+dev) to not have pattern or use always the same examples 
    ds = ds.shuffle()
    # Instances for the 1-shot examples, 500 instances as the test length is 500 
    ds_1S = ds.select(range(500))
    cnt = 0
    for inst in ds_test:
            template = INSTRUCTION + PREFIX_STATEMENT + ds_1S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_1S[cnt-1]['primary_premise'] + '\n' + PREFIX_ANSWER + ds_1S[cnt-1]['label'] + '\n' + SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['primary_premise'] 
            cnt+=1
            all_templates.append(template)
            all_ids.append(inst['id'])
    print("LEN", len(all_templates))

    return all_templates, all_ids


def build_2shot_instances():
    """
        Build all the templates from the test instances in a 2-shot setting, according to our defined template model.
        Corrected examples are taken from the shuffled train and dev sets.

        ADD THE TEMPLATE SCHEME 

        returns: Built template (string)
    """

    all_templates = []
    all_ids = []

    # Load from local files
    og_ds = load_from_disk(PATH_TO_LOCAL_DS)
    ds_train = og_ds['train']
    ds_valid = og_ds['validation']
    ds_test = og_ds['test']
    # Format the premise 
    ds_train = ds_train.map(concat_premise)
    ds_valid = ds_valid.map(concat_premise)
    ds_test = ds_test.map(concat_premise)

    ds = concatenate_datasets([ds_train, ds_valid])
    # Shuffle the dataset (train+dev) to not have pattern or use always the same examples 
    ds = ds.shuffle()
    # Instances for the 1-shot examples, 500 instances as the test length is 500 
    ds_1S = ds.select(range(500))
    ds_2S = ds.select(range(501, 1001))
    cnt = 0
    for inst in ds_test:
            template = INSTRUCTION + PREFIX_STATEMENT + ds_1S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_1S[cnt-1]['primary_premise'] + '\n' + PREFIX_ANSWER + ds_1S[cnt-1]['label'] + '\n' + SEPARATOR_FS + '\n' +INSTRUCTION + PREFIX_STATEMENT + ds_2S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_2S[cnt-1]['primary_premise'] + '\n' + PREFIX_ANSWER + ds_2S[cnt-1]['label'] + '\n' +SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['primary_premise'] 
            all_templates.append(template)
            all_ids.append(inst['id'])
            cnt+=1
    print("LEN", len(all_templates))

    return all_templates, all_ids


### CHAIN OF THOUGHT

def build_1shot_instances_COT():
    """
     
    """
    all_templates = []
    all_ids = []

    # Load from local files
    og_ds = load_from_disk(PATH_TO_LOCAL_DS)
    ds_train = og_ds['train']
    ds_valid = og_ds['validation']
    ds_test = og_ds['test']
    # Format the premise 
    ds_train = ds_train.map(concat_premise)
    ds_valid = ds_valid.map(concat_premise)
    ds_test = ds_test.map(concat_premise)
    # Format the evidence 
    ds_train = ds_train.map(concat_evidence)
    ds_valid = ds_valid.map(concat_evidence)
    ds_test = ds_test.map(concat_evidence)

    ds = concatenate_datasets([ds_train, ds_valid])
    # Shuffle the dataset (train+dev) to not have pattern or use always the same examples 
    ds = ds.shuffle()
    # Instances for the 1-shot examples, 500 instances as the test length is 500 
    ds_1S = ds.select(range(500))
    cnt = 0
    for inst in ds_test:
            template = INSTRUCTION + PREFIX_STATEMENT + ds_1S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_1S[cnt-1]['primary_premise'] + '\n' + PREFIX_EXPLANATION + ds_1S[cnt-1]['primary_evidences'] + MIDDLE_EXPLANATION + ds_1S[cnt-1]['label'] + SUFFIX_EXPLANATION + '\n' + PREFIX_ANSWER + ds_1S[cnt-1]['label'] + '\n' + SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['primary_premise'] 
            cnt+=1
            all_templates.append(template)
            all_ids.append(inst['id'])
    print("LEN", len(all_templates))

    return all_templates, all_ids


def build_2shot_instances_COT():
    """
     TODO doc
    """
    all_templates = []
    all_ids = []

    # Load from local files
    og_ds = load_from_disk(PATH_TO_LOCAL_DS)
    ds_train = og_ds['train']
    ds_valid = og_ds['validation']
    ds_test = og_ds['test']
    # Format the premise 
    ds_train = ds_train.map(concat_premise)
    ds_valid = ds_valid.map(concat_premise)
    ds_test = ds_test.map(concat_premise)
    # Format the evidence 
    ds_train = ds_train.map(concat_evidence)
    ds_valid = ds_valid.map(concat_evidence)
    ds_test = ds_test.map(concat_evidence)

    ds = concatenate_datasets([ds_train, ds_valid])
    # Shuffle the dataset (train+dev) to not have pattern or use always the same examples 
    ds = ds.shuffle()
    # Instances for the 1-shot examples, 500 instances as the test length is 500 
    ds_1S = ds.select(range(500))
    ds_2S = ds.select(range(501, 1001))
    cnt = 0
    for inst in ds_test:
            template = INSTRUCTION + PREFIX_STATEMENT + ds_1S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_1S[cnt-1]['primary_premise'] + '\n' + PREFIX_EXPLANATION + ds_1S[cnt-1]['primary_evidences'] + MIDDLE_EXPLANATION + ds_1S[cnt-1]['label'] + SUFFIX_EXPLANATION + '\n' + PREFIX_ANSWER + ds_1S[cnt-1]['label'] + '\n' + SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + ds_2S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_2S[cnt-1]['primary_premise'] + '\n' + PREFIX_EXPLANATION + ds_2S[cnt-1]['primary_evidences'] + MIDDLE_EXPLANATION + ds_2S[cnt-1]['label'] + SUFFIX_EXPLANATION + '\n' + PREFIX_ANSWER + ds_2S[cnt-1]['label'] + '\n' + SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['primary_premise'] 
            cnt+=1
            all_templates.append(template)
            all_ids.append(inst['id'])
    print("LEN", len(all_templates))

    return all_templates, all_ids


#### CONSTRASTIVE CHAIN OF THOUGHT #### 

def build_1shot_instances_CCOT():
    """
     TODO doc
    """
    all_templates = []
    all_ids = []

    # Load from local files
    og_ds = load_from_disk(PATH_TO_LOCAL_DS)
    ds_train = og_ds['train']
    ds_valid = og_ds['validation']
    ds_test = og_ds['test']
    # Format the wrong evidence, !! to do before the other formatings  
    ds_train = ds_train.map(pick_wrong_evidence)
    ds_valid = ds_valid.map(pick_wrong_evidence)
    ds_test = ds_test.map(pick_wrong_evidence)
    # Format the premise 
    ds_train = ds_train.map(concat_premise)
    ds_valid = ds_valid.map(concat_premise)
    ds_test = ds_test.map(concat_premise)
    # Format the evidence 
    ds_train = ds_train.map(concat_evidence)
    ds_valid = ds_valid.map(concat_evidence)
    ds_test = ds_test.map(concat_evidence)


    ds = concatenate_datasets([ds_train, ds_valid])
    # Shuffle the dataset (train+dev) to not have pattern or use always the same examples 
    ds = ds.shuffle()
    # Instances for the 1-shot examples, 500 instances as the test length is 500 
    ds_1S = ds.select(range(500))
    cnt = 0
    for inst in ds_test:
            template = INSTRUCTION + PREFIX_STATEMENT + ds_1S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_1S[cnt-1]['primary_premise'] + '\n' + PREFIX_EXPLANATION + ds_1S[cnt-1]['primary_evidences'] + MIDDLE_EXPLANATION + ds_1S[cnt-1]['label'] + SUFFIX_EXPLANATION + '\n' + PREFIX_WRONG_EXPLANATION + ds_1S[cnt-1]['wrong_evidences'] + MIDDLE_WRONG_EXPLANATION + ds_1S[cnt-1]['label'] + SUFFIX_WONRG_EXPLANATION+ '\n' + PREFIX_ANSWER + ds_1S[cnt-1]['label'] + '\n' + SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['primary_premise'] 
            cnt+=1
            all_templates.append(template)
            all_ids.append(inst['id'])
    print("LEN", len(all_templates))

    return all_templates, all_ids


def build_2shot_instances_CCOT():
    """
        TODO doc
    """
    all_templates = []
    all_ids = []

    # Load from local files
    og_ds = load_from_disk(PATH_TO_LOCAL_DS)
    ds_train = og_ds['train']
    ds_valid = og_ds['validation']
    ds_test = og_ds['test']
    # Format the wrong evidence, !! to do before the other formatings  
    ds_train = ds_train.map(pick_wrong_evidence)
    ds_valid = ds_valid.map(pick_wrong_evidence)
    ds_test = ds_test.map(pick_wrong_evidence)
    # Format the premise 
    ds_train = ds_train.map(concat_premise)
    ds_valid = ds_valid.map(concat_premise)
    ds_test = ds_test.map(concat_premise)
    # Format the evidence 
    ds_train = ds_train.map(concat_evidence)
    ds_valid = ds_valid.map(concat_evidence)
    ds_test = ds_test.map(concat_evidence)


    ds = concatenate_datasets([ds_train, ds_valid])
    # Shuffle the dataset (train+dev) to not have pattern or use always the same examples 
    ds = ds.shuffle()
    # Instances for the 1-shot examples, 500 instances as the test length is 500 
    ds_1S = ds.select(range(500))
    ds_2S = ds.select(range(501, 1001))
    cnt = 0
    for inst in ds_test:
            template = INSTRUCTION + PREFIX_STATEMENT + ds_1S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_1S[cnt-1]['primary_premise'] + '\n' + PREFIX_EXPLANATION + ds_1S[cnt-1]['primary_evidences'] + MIDDLE_EXPLANATION + ds_1S[cnt-1]['label'] + SUFFIX_EXPLANATION + '\n' + PREFIX_WRONG_EXPLANATION + ds_1S[cnt-1]['wrong_evidences'] + MIDDLE_WRONG_EXPLANATION + ds_1S[cnt-1]['label'] + SUFFIX_WONRG_EXPLANATION+ '\n' + PREFIX_ANSWER + ds_1S[cnt-1]['label'] + '\n' + SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + ds_2S[cnt-1]['statement'] + '\n' + PREFIX_PREMISE + ds_2S[cnt-1]['primary_premise'] + '\n' + PREFIX_EXPLANATION + ds_2S[cnt-1]['primary_evidences'] + MIDDLE_EXPLANATION + ds_2S[cnt-1]['label'] + SUFFIX_EXPLANATION + '\n' + PREFIX_ANSWER + ds_2S[cnt-1]['label'] + '\n' + PREFIX_WRONG_EXPLANATION + ds_2S[cnt-1]['wrong_evidences'] + MIDDLE_WRONG_EXPLANATION + ds_2S[cnt-1]['label'] + SUFFIX_WONRG_EXPLANATION+ '\n' + SEPARATOR_FS + '\n' + INSTRUCTION + PREFIX_STATEMENT + inst['statement'] + '\n' + PREFIX_PREMISE + inst['primary_premise'] 
            cnt+=1
            all_templates.append(template)
            all_ids.append(inst['id'])
    print("LEN", len(all_templates))

    return all_templates, all_ids

build_2shot_instances_CCOT()
