import os
import json
import transformers
import datasets

from datasets import load_dataset, DatasetDict

# bigbio/sem_eval_2024_task_2

#train_dataset = datasets.load_dataset("bigbio/sem_eval_2024_task_2", split="validation")

#print(train_dataset[14])

CTR_FOLDER = "/home/aguiar/Documents/SE_2024_Task_2/training_data/training_data/CT_json"
TRAIN_FILE = "/home/aguiar/Documents/SE_2024_Task_2/training_data/training_data/train.json"
VALID_FILE = "/home/aguiar/Documents/SE_2024_Task_2/training_data/training_data/dev.json"

def fetch_evidences(primary_id, secondary_id):
    """
        Fetch the evidence index from the dev.json and train.json 
    """
    primary_evidences_idx = None
    secondary_evidences_idx = None
    
    # Look if the CTR is in the dev set
    if primary_id is not None:
        with open(VALID_FILE, 'r') as f:
            content_valid_file = json.load(f)
        # print(content_valid_file)
        for id in content_valid_file:
            #print(id)
            if content_valid_file[id]['Primary_id'] == primary_id:
                primary_evidences_idx = content_valid_file[id]['Primary_evidence_index']
                print("AA", primary_evidences_idx)

        # Check if we need to test within the train file 
        if primary_evidences_idx is None:
            with open(TRAIN_FILE, 'r') as f:
                content_train_file = json.load(f)
            # print(content_valid_file)
            for id in content_train_file:
                #print(id)
                if content_train_file[id]['Primary_id'] == primary_id:
                    primary_evidences_idx = content_train_file[id]['Primary_evidence_index']
                    print("BB", primary_evidences_idx)
    
    if secondary_id is not None:
        with open(VALID_FILE, 'r') as f:
            content_valid_file = json.load(f)
        # print(content_valid_file)
        for id in content_valid_file:
            # print(id)
            if 'Secondary_id' in content_valid_file[id] and content_valid_file[id]['Secondary_id'] == secondary_id:
                secondary_evidences_idx = content_valid_file[id]['Secondary_evidence_index']
                print("CC", secondary_evidences_idx)

        # Check if we need to test within the train file 
        if secondary_evidences_idx is None:
            with open(TRAIN_FILE, 'r') as f:
                content_train_file = json.load(f)
            # print(content_valid_file)
            for id in content_train_file:
                #print(id)
                if 'Secondary_id' in content_train_file[id] and content_train_file[id]['Secondary_id'] == secondary_id:
                    secondary_evidences_idx = content_train_file[id]['Secondary_evidence_index']
                    print("DD", secondary_evidences_idx)

    return primary_evidences_idx, secondary_evidences_idx


#fetch_evidences('NCT00003404','NCT00711529')


def fetch_ctr(primary_id, secondary_id, section_id):


    """
        Function to fetch the text of the section of a CTR where the premise is. If this is a single type of trial (there is no secondary CTR to take into account), 
        we fetch only the section of the primary trial following respectively section_id and primary_id to get the desired section. 
        If this is not a single type of trial, there are 2 CTR involved, so we fetch both sections of the 2 CTRs corresponding to secondary_id and primary_id. 

        TODO add the thing about evidences 

        :primary_id: string
        :secondary_id: string
        :section_id: string

        returns: premise from primary trial, premise from secondary trial 
    """
    primary_path = None
    secondary_path = None
    primary_premise = None
    secondary_premise = None
    primary_evidences_idx = None
    secondary_evidences_idx = None
    primary_evidences = []
    secondary_evidences = []

    primary_evidences_idx, secondary_evidences_idx = fetch_evidences(primary_id, secondary_id)

    if primary_id is not None:
        for root, dirs, files in os.walk(CTR_FOLDER):
            if primary_id+'.json' in files:
                primary_path =  os.path.join(root, primary_id+'.json')

        if primary_path is not None:    
            with open(primary_path, 'r') as f:
                content_ctr = json.load(f)
            primary_premise = content_ctr[section_id]
            # get the CTR name --> look for the corresponding name among the evidence dict 
            # get the corresponding array of evidence idx 
            if primary_evidences_idx is not None:
                primary_evidences = [primary_premise[idx] for idx in primary_evidences_idx]
                print("PRIM EVID", primary_evidences)
            # print("PRIMARY PREMISE", primary_premise)

    if secondary_id is not None:
        for root, dirs, files in os.walk(CTR_FOLDER):
            if secondary_id+'.json' in files:
                secondary_path =  os.path.join(root, secondary_id+'.json')
        if secondary_path is not None:
            with open(secondary_path, 'r') as f:
                content_ctr = json.load(f)
            secondary_premise  = content_ctr[section_id]

            if secondary_evidences_idx is not None:
                secondary_evidences = [secondary_premise[idx] for idx in secondary_evidences_idx]
                print("SECOND EVID", secondary_evidences)
   
            # print("SECONDARY PREMISE", secondary_premise)

    return primary_premise, secondary_premise, primary_evidences, secondary_evidences



def add_premises_to_dataset():
    """
    This function shaped the premise into a regular text and add new column to the BigBio's version of the SemEval dataset. 
    :primary_premise: extracted primary premise 
    :secondary_premise: extracted secondary premise 
    :semeval_ds: Bigbio's semeval dataset 

    returns: modified dataset
    
    """
    # TODO ADD THE COLUMN WITH THE EVIDENCE SENTENCES 

    all_secondary_premise_train = []
    all_primary_premise_train = []
    all_primary_evidences_train = []
    all_secondary_evidences_train = []
    # all_primary_evidences_idx_train = []
    # all_secondary_evidences_idx_train = []
    all_secondary_premise_valid = []
    all_primary_premise_valid = []
    all_primary_evidences_valid = []
    all_secondary_evidences_valid = []
    # all_primary_evidences_idx_valid = []
    # all_secondary_evidences_idx_valid = []

    og_train_dataset = load_dataset("bigbio/sem_eval_2024_task_2", split='train') 
    for inst in og_train_dataset:
        primary_premise, secondary_premise, primary_evidences, secondary_evidences = fetch_ctr(inst['primary_id'], inst['secondary_id'], inst['section_id'])
        all_primary_premise_train.append(primary_premise)
        all_secondary_premise_train.append(secondary_premise)
        all_primary_evidences_train.append(primary_evidences)
        all_secondary_evidences_train.append(secondary_evidences)
    ds_train = og_train_dataset.add_column("primary_premise", all_primary_premise_train)
    print(ds_train)
    ds_train = ds_train.add_column("secondary_premise", all_secondary_premise_train)
    ds_train = ds_train.add_column("primary_evidences", all_primary_evidences_train)
    ds_train = ds_train.add_column("secondary_evidences", all_secondary_evidences_train)

 
    og_valid_dataset = load_dataset("bigbio/sem_eval_2024_task_2", split='validation')
    for inst in og_valid_dataset:
        primary_premise, secondary_premise = fetch_ctr(inst['primary_id'], inst['secondary_id'], inst['section_id'])
        #ds['validation']['primary_prepmise'] = primary_premise
        #ds['validation']['secondary_premise'] = secondary_premise
        all_primary_premise_valid.append(primary_premise)
        all_secondary_premise_valid.append(secondary_premise)
        all_primary_evidences_valid.append(primary_evidences)
        all_secondary_evidences_valid.append(secondary_evidences)
    ds_valid = og_valid_dataset.add_column("primary_premise", all_primary_premise_valid)
    ds_valid = ds_valid.add_column("secondary_premise", all_secondary_premise_valid)
    ds_valid = ds_valid.add_column("primary_evidences", all_primary_evidences_valid)
    ds_valid = ds_valid.add_column("secondary_evidences", all_secondary_evidences_valid)   

    # Merge train and valid subsets
    ds = DatasetDict({
        "train": ds_train,
        "validation": ds_valid
    })
    print(ds['validation']['secondary_evidences'])
    # ds.save_to_disk("test")
    return ds


add_premises_to_dataset()


#fetch_ctr('NCT00499083','NCT03045653','Eligibility')