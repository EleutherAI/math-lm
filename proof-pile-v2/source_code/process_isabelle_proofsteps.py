import os
import glob
import json
from tqdm import tqdm
import re
import argparse

def split_list(lst):
        splits = []
        temp = []
        for item in lst:
            if item[0] == '':
                if temp:
                    splits.append(temp)
                temp = [item]
            else:
                temp.append(item)
        if temp:
            splits.append(temp)
        return splits

def create_outputs(split):
    res = ""

    statement = split[0][1]

    res += "[STATEMENT]\n" + statement

    for i in range(1, len(split)):
        state = split[i][0]
        step = split[i][1]
        res += "\n[PROOF STATE]\n" + state + "\n"
        res += "[PROOF STEP]\n" + step

    length = len(split) - 1

    return res, length

def extract_theory_file(json_filename):
    # Regex to match the text between '2022-12-06_thys_' and '.thy_output.json'
    pattern = r'2022-12-06_thys_(.*?)\.thy_output\.json'
    match = re.search(pattern, json_filename)
    if match:
        return match.group(1)
    return None

def get_theorem_statements_from_folder(folder_path):
    theorem_statements = []
    json_files_in_folder = glob.glob(os.path.join(folder_path, '*.json'))

    for json_file in json_files_in_folder:
        with open(json_file, 'r') as f:
            data = json.load(f)

            for item in data:
                if len(item) > 1:
                    theorem_statements.append(item[1])

    return theorem_statements

def create_dataset(path_to_dataset, test_set, decontaminate=True):
    dataset = []

    json_files = glob.glob(path_to_dataset + '/*.json')
    json_files = [file for file in json_files if file.endswith('thy_output.json')]

    all_splits = []
    corresponding_files = []

    for file in tqdm(json_files):
        with open(file) as f:
            data = json.load(f)
        if 'translations' in data:
            translations = data['translations']
            splitted_list = split_list(translations)
            splitted_list = [x for x in splitted_list if x[0][1].startswith('lemma ') or x[0][1].startswith('theorem ')]
            for split in splitted_list:
                all_splits.append(split)
                corresponding_files.append(file)

    for i in tqdm(range(len(all_splits))):
        split = all_splits[i]
        json_file = corresponding_files[i]
        txt, length = create_outputs(split)
        file = extract_theory_file(json_file)
        if decontaminate and any(theorem in txt for theorem in test_set):
            continue
        else:
            dataset.append({"proof": txt, "metadata": {"file": file, "length": length}})

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--afp_folder", help="Path to the AFP folder")
    parser.add_argument("--std_folder", help="Path to the STD folder")
    parser.add_argument("--test_folder", help="Path to the PISA test set folder")
    parser.add_argument("--results_file", default='pisa_dataset.jsonl', help="Path to the results file")
    args = parser.parse_args()

    test_set = get_theorem_statements_from_folder(args.test_folder)
    afp_dataset_decontaminated = create_dataset(args.afp_folder, test_set)
    std_dataset_decontaminated = create_dataset(args.std_folder, test_set)

    with open(args.results_file, "a") as file:
        for data in tqdm(afp_dataset_decontaminated):
            json_str = json.dumps(data)
            file.write(json_str + "\n")
        for data in tqdm(std_dataset_decontaminated):
            json_str = json.dumps(data)
            file.write(json_str + "\n")