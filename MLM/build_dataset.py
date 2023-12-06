import os
import json
import transformers
import datasets

from datasets import load_dataset, DatasetDict, Dataset

# bigbio/sem_eval_2024_task_2

#train_dataset = datasets.load_dataset("bigbio/sem_eval_2024_task_2", split="validation")

#print(train_dataset[14])

CTR_FOLDER = "/home/aguiar/Documents/SE_2024_Task_2/SemEval-2024-Task-2/training_data/CT_json"
TRAIN_FILE = "/home/aguiar/Documents/SE_2024_Task_2/SemEval-2024-Task-2/training_data/train.json"
VALID_FILE = "/home/aguiar/Documents/SE_2024_Task_2/SemEval-2024-Task-2/training_data/dev.json"
PRACTICE_TEST_FILE = "/home/aguiar/Documents/SE_2024_Task_2/SemEval-2024-Task-2/training_data/practice_test.json"

def fetch_evidences(primary_id, secondary_id, split):
    """
        Fetch the evidence index from the dev.json and train.json 
    """
    primary_evidences_idx = None
    secondary_evidences_idx = None
    
    # Look if the CTR is in the dev set
    if primary_id is not None:
        if split == 'validation':
            with open(VALID_FILE, 'r') as f:
                content_valid_file = json.load(f)
            # print(content_valid_file)
            for id in content_valid_file:
                #print(id)
                if content_valid_file[id]['Primary_id'] == primary_id:
                    primary_evidences_idx = content_valid_file[id]['Primary_evidence_index']
                    #print("AA", primary_evidences_idx)

        # Check if we need to test within the train file 
        if split == 'train':  # primary_evidences_idx is None
            with open(TRAIN_FILE, 'r') as f:
                content_train_file = json.load(f)
            # print(content_valid_file)
            for id in content_train_file:
                #print(id)
                if content_train_file[id]['Primary_id'] == primary_id:
                    primary_evidences_idx = content_train_file[id]['Primary_evidence_index']
                    #print("BB", primary_evidences_idx)
    
    if secondary_id is not None:
        if split == 'validation':
            with open(VALID_FILE, 'r') as f:
                content_valid_file = json.load(f)
            # print(content_valid_file)
            for id in content_valid_file:
                # print(id)
                if 'Secondary_id' in content_valid_file[id] and content_valid_file[id]['Secondary_id'] == secondary_id:
                    secondary_evidences_idx = content_valid_file[id]['Secondary_evidence_index']
                    #print("CC", secondary_evidences_idx)

        # Check if we need to test within the train file 
        if split == 'train':  # secondary_evidences_idx is None
            with open(TRAIN_FILE, 'r') as f:
                content_train_file = json.load(f)
            # print(content_valid_file)
            for id in content_train_file:
                #print(id)
                if 'Secondary_id' in content_train_file[id] and content_train_file[id]['Secondary_id'] == secondary_id:
                    secondary_evidences_idx = content_train_file[id]['Secondary_evidence_index']
                    #print("DD", secondary_evidences_idx)

    return primary_evidences_idx, secondary_evidences_idx


#fetch_evidences('NCT00003404','NCT00711529')


def fetch_ctr(primary_id, secondary_id, section_id, split):


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

    if split == 'train' or split == 'validation':
        primary_evidences_idx, secondary_evidences_idx = fetch_evidences(primary_id, secondary_id, split)

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
                # print("PRIM EVID", primary_evidences)
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
                # print("SECOND EVID", secondary_evidences)
   
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
    all_secondary_premise_test = []
    all_primary_premise_test = []

    og_train_dataset = load_dataset("bigbio/sem_eval_2024_task_2", split='train') 
    for inst in og_train_dataset:
        primary_premise, secondary_premise, primary_evidences, secondary_evidences = fetch_ctr(inst['primary_id'], inst['secondary_id'], inst['section_id'], 'train')
        all_primary_premise_train.append(primary_premise)
        all_secondary_premise_train.append(secondary_premise)
        all_primary_evidences_train.append(primary_evidences)
        all_secondary_evidences_train.append(secondary_evidences)
    ds_train = og_train_dataset.add_column("primary_premise", all_primary_premise_train)
    # print(ds_train)
    ds_train = ds_train.add_column("secondary_premise", all_secondary_premise_train)
    ds_train = ds_train.add_column("primary_evidences", all_primary_evidences_train)
    ds_train = ds_train.add_column("secondary_evidences", all_secondary_evidences_train)

 
    og_valid_dataset = load_dataset("bigbio/sem_eval_2024_task_2", split='validation')
    for inst in og_valid_dataset:
        primary_premise, secondary_premise, primary_evidences, secondary_evidences = fetch_ctr(inst['primary_id'], inst['secondary_id'], inst['section_id'], 'validation')
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

    # Integration of the Practice test set 

    with open(PRACTICE_TEST_FILE) as json_file:
        json_practice_test = json.load(json_file)
    #print("XXXX", json_practice_test)

    #json_practice_test = load_dataset('json', data_files=PRACTICE_TEST_FILE)
    cleaned_ds = {}
    all_ids_test = []
    all_labels_test = []
    all_statements_test = []
    all_type_test = []
    all_section_id_test = []
    all_primary_id_test = []
    all_secondary_id_test = []

    for inst in json_practice_test:
        #print('YYY', inst)
        #print("ZZZ", json_practice_test[inst])
        #print('ZZ', json_practice_test[idx_inst]['Primary_id'])
        secondary_id = None
        if 'Secondary_id' in json_practice_test[inst].keys() is not None:
            secondary_id = json_practice_test[inst]['Secondary_id']
        else:
            json_practice_test[inst]['Secondary_id'] = None

        primary_premise, secondary_premise, primary_evidences, secondary_evidences = fetch_ctr(json_practice_test[inst]['Primary_id'], secondary_id, json_practice_test[inst]['Section_id'], 'test')
        #print("aaaaaa", primary_premise)
        #print("bbbb", secondary_premise)
        all_primary_premise_test.append(primary_premise)
        all_secondary_premise_test.append(secondary_premise)
        # To correct the mismatches of the original json file 
        if 'Label' not in json_practice_test[inst].keys():
            json_practice_test[inst]['Label'] = None
        all_labels_test.append(json_practice_test[inst]['Label'])
        all_ids_test.append(str(inst))
        all_statements_test.append(json_practice_test[inst]['Statement'])
        all_type_test.append(json_practice_test[inst]['Type'])
        all_secondary_id_test.append(json_practice_test[inst]['Secondary_id'])
        all_primary_id_test.append(json_practice_test[inst]['Primary_id'])
        all_section_id_test.append(json_practice_test[inst]['Section_id'])
        # add the new columns manually
        
        #print(dict(json_practice_test[inst]))
        #cleaned_ds.append(dict(json_practice_test[inst]))
        #print(cleaned_ds)
    cleaned_ds = {
        'id': all_ids_test,
        'type': all_type_test,
        'section_id': all_section_id_test,
        'primary_id': all_primary_id_test,
        'secondary_id': all_secondary_id_test,
        'statement': all_statements_test,
        'label': all_labels_test,
        'primary_premise': all_primary_premise_test,
        'secondary_premise': all_secondary_premise_test,
    }
    # TODO le json_practice_test est un dict et pas un dataset 
    #print("DDDD", json_practice_test)
    ds_test = Dataset.from_dict(cleaned_ds)
    print("CACAAAA", ds_test)
    #print('KKKK', ds_test)
    #print('kkk', len(all_secondary_premise_test))
    #ds_test = ds_test.add_column("primary_premise", all_primary_premise_test)
    #ds_test = ds_test.add_column("secondary_premise", all_secondary_premise_test)

    # Merge train and valid subsets
    ds = DatasetDict({
        "train": ds_train,
        "validation": ds_valid,
        "test": ds_test
    })
    print(ds)
    ds.save_to_disk("/home/aguiar/Documents/SE_2024_Task_2/SemEval-2024-Task-2/training_data/dump")
    return ds


add_premises_to_dataset()


#fetch_ctr('NCT00499083','NCT03045653','Eligibility')