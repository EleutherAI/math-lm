from datasets import load_dataset
import re
import random
import fire
import json
random.seed(2007)


LEVEL_COUNTS = {'Level 5': 300, 'Level 4': 165, 'Level 3': 150}

def remove_boxed(text):
    pattern = r'\\boxed{(.*)}'
    return re.sub(pattern, r'\1', text)

def fix_align(text):
    pattern = r'\\begin{align\*}(.*?)\\end{align\*}'
    text = re.sub(
        pattern, 
        '\n$$\n\\\\begin{align*}\t\\1\\\\end{align*}\n$$\n', 
        text, flags=re.DOTALL
    )
    
    return text

def convert_latex(text):
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', '\n$$\n\t\\1\n$$\n', text, flags=re.DOTALL)

    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    text = re.sub(r'\$\$\$\$', r'$$\n$$', text)

    return text

fix = lambda x: convert_latex(fix_align(remove_boxed(x)))

def blacklist(x):
    if "[asy]" in x["solution"]:
        return True
    elif "eqnarray" in x["solution"]:
        return True
    elif "tabular" in x["solution"]:
        return True
    else:
        return False

def download(destpath):
    math = load_dataset("hendrycks/competition_math")

    data = [
        {
            "input": x["problem"], 
            "output": fix(x["solution"]),
            "meta": {"level": x["level"], "type": x["type"], "id": 10**8+i}
        }
        for i, x in enumerate(math["train"]) if not blacklist(x)
    ]

    data_by_level = dict()
    for k,v in LEVEL_COUNTS.items():
        level_data = [x for x in data if x['meta']['level']==k]
        data_by_level[k] = random.sample(level_data, v)
    
    post_data = [x for k,v in data_by_level.items() for x in v]

    with open(destpath, "w") as f:
        [f.write(json.dumps(x) + "\n") for x in post_data]

if __name__=="__main__":
    fire.Fire(download)