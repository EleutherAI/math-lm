import os
import json
import ndjson
import argparse

matlab_meta_keys = ['num_repos', 'num_examples', 'num_tokens']

def add_meta_dicts(x1, x2):
    x1_m = x1['MATLAB']
    x2_m = x2['MATLAB']
    return {
            'MATLAB': {k: x1_m[k] + x2_m[k] for k in matlab_meta_keys}
    }

def main(args):
    splits = ['train', 'validation', 'test']

    ds_dict = {k: [] for k in splits}
    meta_dict = {'MATLAB': {'num_repos': 0, 'num_examples': 0, 'num_tokens': 0}}

    for year in range(args.first_year, args.last_year+1):
        print(f"processing year {year}...")
        for split in splits:
            base_dir = os.path.join(args.matlab_by_year_dir, f'data_jsonl{year}', split)
            for filename in os.listdir(base_dir):
                if filename.endswith('.jsonl'):
                    filepath = os.path.join(base_dir, filename)

                    with open(filepath) as f:
                        ds_dict[split] += ndjson.load(f)

        with open(os.path.join(args.matlab_by_year_dir, f'meta_json{year}', 'github-stats.json')) as f:
            meta_dict = add_meta_dicts(meta_dict, json.load(f))

    print(f'total train documents: {len(ds_dict["train"])}')

    for split in splits:
        print(f'saving {split}')
        to_save = ds_dict[split]
        for i, left in enumerate(range(0, len(to_save), args.shard_size)):
            print('saving shard {i}')
            batch = to_save[left:left+args.shard_size]

            with open(os.path.join(args.destination_dir, split, f'github-MATLAB-{split}-{str(i).zfill(4)}.jsonl'), 'w') as f:
                ndjson.dump(batch, f)
    
    meta_path = os.path.join(args.destination_meta_dir, 'github-stats.json')
    with open(meta_path) as f:
        current_meta = json.load(f)

    merged_meta = {**current_meta, **meta_dict}

    with open(meta_path, 'w') as f:
        json.dump(merged_meta, f)


if __name__=="__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument('--first_year', type=int, default=2009)
    parser.add_argument('--last_year', type=int, default=2021)
    parser.add_argument('--matlab_by_year_dir', type=str, default='matlab_by_year')
    parser.add_argument('--destination_dir', type=str, default='data_jsonl')
    parser.add_argument('--destination_meta_dir', type=str, default='meta_json')
    parser.add_argument('--shard_size', type=int, default=50_000)

    args = parser.parse_args()

    main(args)
