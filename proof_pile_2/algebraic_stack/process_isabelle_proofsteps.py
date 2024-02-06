import os
import glob
import json
from tqdm import tqdm
import re
import argparse
import textwrap

import code

def wrap_isabelle_comment(text, width=80):
    # Constants for comment formatting

    if "\n" not in text:
        return f"(* {text} *)"

    wrapped_text = ""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if i==0:
            wrapped_text += "(*  " + line + "\n"
        elif i==len(lines)-1:
            wrapped_text += "    " + line + "\n*)"
        else:
            wrapped_text += "    " + line + "\n"
    
    # Wrap the text

    return wrapped_text


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

    res += statement

    for i in range(1, len(split)):
        state = split[i][0]
        step = split[i][1]
        res += "\n" + wrap_isabelle_comment(state) + "\n"
        res += step

    length = len(split) - 1


    return res, length

def extract_filename(filepath):
    begin = "thys_"
    end = "_ground"
    whole_name = filepath[filepath.index(begin) + len(begin):filepath.index(end)]

    theory = whole_name[:whole_name.rindex("_")]
    filename = whole_name[whole_name.rindex("_")+1:]

    return theory + "/" + filename + ".thy"

def consolidate_by_file(data):
    row_of_file = dict()

    for row in tqdm(data):
        filename = row["meta"]["file"]
        if filename not in row_of_file:
            row_of_file[filename] = [row]
        else:
            row_of_file[filename].append(row)
    
    consolidated_data = []
    for k in tqdm(row_of_file):
        v = row_of_file[k]
        # code.interact(local=locals())
        text = "\n\n".join([x["text"] for x in v])
        consolidated_data.append(
            {"text": text, "meta": {"file": v[0]["meta"]["file"]}}
            )

    return consolidated_data



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

def create_dataset(path_to_dataset, test_set, decontaminate=True, max_items=None):
    dataset = []

    json_files = glob.glob(path_to_dataset + '/*/*.json')
    json_files = [file for file in json_files if True] # file.endswith('thy_output.json')]

    all_splits = []
    corresponding_files = []

    if max_items:
        json_files = json_files[:max_items]

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
        file = extract_filename(json_file)
        if decontaminate and any(theorem in txt for theorem in test_set):
            continue
        else:
            dataset.append({"text": txt, "meta": {"file": file, "length": length}})

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--afp_folder", help="Path to the AFP folder")
    # parser.add_argument("--std_folder", help="Path to the STD folder")
    parser.add_argument("--test_folder", help="Path to the PISA test set folder")
    parser.add_argument("--results_file", default='pisa_dataset.jsonl', help="Path to the results file")
    parser.add_argument("--max_items", default=None, type=int, help="Maximum jsons to process")
    args = parser.parse_args()

    test_set = get_theorem_statements_from_folder(args.test_folder)
    afp_dataset_decontaminated = create_dataset(args.afp_folder, test_set, max_items=args.max_items)
    # std_dataset_decontaminated = create_dataset(args.std_folder, test_set)

    afp_dataset_decontaminated = consolidate_by_file(afp_dataset_decontaminated)

    with open(args.results_file, "w") as file:
        for data in tqdm(afp_dataset_decontaminated):
            json_str = json.dumps(data)
            file.write(json_str + "\n")
        # for data in tqdm(std_dataset_decontaminated):
        #     json_str = json.dumps(data)
        #     file.write(json_str + "\n")
