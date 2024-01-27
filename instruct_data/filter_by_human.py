import json
import yaml
import fire

def filter_by_human(inputpath: str, destpath: str, judgement_file:str):
    with open(judgement_file) as fle:
        judgement_of_id = yaml.safe_load(fle)
    
    good_ids = set([k for k,v in judgement_of_id.items() if v])

    with open(inputpath) as fle:
        pre_data = [json.loads(x) for x in fle]
    
    post_data = filter(lambda x: x["meta"]["post_id"] in good_ids, pre_data)

    with open(destpath, 'w') as fle:
        for row in post_data:
            fle.write(json.dumps(row) + '\n')

if __name__=="__main__":
    fire.Fire(filter_by_human)



