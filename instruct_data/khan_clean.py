import json
from typing import Dict
import re
from tqdm import tqdm
import fire
import os
import code

from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI 
import tiktoken

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

ENDPOINT="gpt-3.5-turbo-1106"

build_prompt = lambda input, output: f"""\
The mathematical content of the problem and solution below is correct. \
However, the exposition and formatting might be suboptimal. \
If necessary, please rewrite the problem and solution \
so that they are well-written and in readable Markdown LaTeX.

# Question
{input}

# Solution
{output}

Format your response as 
# Question
[your rewritten question goes here]
# Solution
[your rewrittens solution goes here]"""

def extract_spans(text):
    pattern = r'#+\s*Question\n(.*?)#+\s*Solution\n(.*)'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        span_1, span_2 = match.groups()
        return span_1, span_2
    else:
        return None, None

def convert_latex(text):
    # Convert display math mode
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

    # Convert inline math mode
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    return text


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def complete(client, prompt):
    return client.chat.completions.create(
        model=ENDPOINT,
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful mathematical assistant."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        top_p=0.95,
    )


def paraphrase_of_example(client, destpath, row: Dict, lock):
    raw_input = row["input"]
    raw_output = row["output"]
    new = {
        "meta": {
            "raw_input": raw_input,
            "raw_output": raw_output,
            **row["meta"],
        }
    }

    prompt = build_prompt(raw_input, raw_output)
    completion = complete(client, prompt)

    # code.interact(local=locals())

    text = completion.choices[0].message.content
    
    q, a = extract_spans(text)

    if not q:
        q = ""
    if not a:
        a = ""

    # code.interact(local=locals())

    new["input"] = convert_latex(q)
    new["output"] = convert_latex(a)
    new["meta"]["openai_response"] = completion.model_dump()

    with lock:
        with open(destpath, "a") as f:
            f.write(json.dumps(new) + "\n")


def paraphrase_dataset(inputpath, destpath, max_rows=None):
    if os.path.isfile(destpath):
        raise OSError("destpath already exists")

    client = OpenAI()

    with open(inputpath) as f:
        data = [json.loads(x) for x in f]

    if max_rows:
        data = data[:max_rows]

    executor = ThreadPoolExecutor(max_workers=50)
    lock = Lock()
    futures = []

    for x in data:
        future = executor.submit(
            paraphrase_of_example, client, destpath, x, lock
        )
        futures.append(future)

    with tqdm(total=len(futures)) as progress:
        for future in as_completed(futures):
            progress.update(1)

    executor.shutdown(wait=True)


if __name__=="__main__":
    fire.Fire(paraphrase_dataset)