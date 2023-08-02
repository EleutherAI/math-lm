"""This code is adapted from:

@article{liu2023goat,
  title={Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks},
  author={Liu, Tiedong and Low, Bryan Kian Hsiang},
  journal={arXiv preprint arXiv:2305.14201},
  year={2023}
}
https://github.com/liutiedong/goat
"""

import json
import os
import random
import tiktoken
from pathlib import Path
from tqdm import trange


def addition(k=1000):
    # Addition up to 16 digits
    pairs = \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(1, 16)
         for j in range(i, 16)
         for _ in range(k)] + \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(3, 16)
         for j in range(i, 16)
         for _ in range(k)] + \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(6, 16)
         for j in range(i, 16)
         for _ in range(k)] + \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(9, 16)
         for j in range(i, 16)
         for _ in range(k)] + \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(12, 16)
         for j in range(i, 16)
         for _ in range(k)]

    random.shuffle(pairs)

    print("Addition:", len(pairs))
    data = []
    for num1, num2 in pairs:
        if random.random() < 0.5:
            num1, num2 = num2, num1
        answer = num1 + num2
        question = f"{num1} + {num2}"
        output = f"{num1} + {num2} = {answer}"
        assert (output.split()[-1] == str(answer))
        data.append({"input": question, "output": output, "answer": str(answer)})
    return data


def subtraction(k=1000):
    # Subtraction up to 16 digits
    pairs = \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(1, 16)
         for j in range(i, 16)
         for _ in range(k)] + \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(3, 16)
         for j in range(i, 16)
         for _ in range(k)] + \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(6, 16)
         for j in range(i, 16)
         for _ in range(k)] + \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(9, 16)
         for j in range(i, 16)
         for _ in range(k)] + \
        [(random.randint(10 ** (i - 1), 10 ** i), random.randint(10 ** (j - 1), 10 ** j))
         for i in range(12, 16)
         for j in range(i, 16)
         for _ in range(k)]

    random.shuffle(pairs)

    print("Subtraction:", len(pairs))
    data = []
    for num1, num2 in pairs:
        if random.random() < 0.5:
            num1, num2 = num2, num1

        answer = num1 - num2
        question = f"{num1} - {num2}"
        output = f"{num1} - {num2} = {answer}"
        assert (output.split()[-1] == str(answer))
        data.append({"input": question, "output": output, "answer": str(answer)})
    return data


def mul_1_n(k=1000):
    # 1xn, up to 16 digits.
    pairs = \
        [(random.randint(2, 9), random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(2, 5)
         for _ in range(4*k)] + \
        [(random.randint(2, 9), random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(5, 8)
         for _ in range(8*k)] + \
        [(random.randint(2, 9), random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(8, 12)
         for _ in range(12*k)] + \
        [(random.randint(2, 9), random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(12, 17)
         for _ in range(16*k)] + \
        [(0, random.randint(10 ** (j - 1) + 1, 10 ** j - 1))
         for j in range(2, 16)
         for _ in range(k//2)] + \
        [(1, random.randint(10 ** (j - 1) + 1, 10 ** j - 1))
         for j in range(2, 16)
         for _ in range(k//2)] + \
        [(10, random.randint(10 ** (j - 1) + 1, 10 ** j - 1))
         for j in range(2, 16)
         for _ in range(k//2)] + \
        [(random.randint(1, 9), random.randint(1, 9))
         for _ in range(k//2)]

    random.shuffle(pairs)

    print("Multiplication nx1:", len(pairs))

    data = []

    for num1, num2 in pairs:
        if random.random() < 0.1:
            num1 = num1 * (10 ** random.randint(1, 6))

        if random.random() < 0.5:
            num1, num2 = num2, num1

        answer = num1 * num2
        question = f"{num1} * {num2}"
        output = f"{num1} * {num2} = {answer}"

        assert (output.split()[-1] == str(answer))
        data.append({"input": question, "output": output, "answer": str(answer)})
    return data


def mul_n_m(k=1000):
    # multi-digit multiplication, with the product up to 12 digits
    pairs = \
        [(random.randint(10 ** (i - 1) + 1, 10 ** i - 1), random.randint(10 ** (j - 1) + 1, 10 ** j - 1))
         for i in range(2, 7)
         for j in range(i, 13 - i)
         for _ in range(10*k)] + \
        [(random.randint(10 ** (i - 1) + 1, 10 ** i - 1), random.randint(10 ** (j - 1) + 1, 10 ** j - 1))
         for i in range(4, 7)
         for j in range(i, 13 - i)
         for _ in range(10*k)] + \
        [(random.randint(10 ** (i - 1) + 1, 10 ** i - 1), random.randint(10 ** (j - 1) + 1, 10 ** j - 1))
         for i in range(5, 7)
         for j in range(i, 13 - i)
         for _ in range(10*k)] + \
        [(random.randint(10 ** (i - 1) + 1, 10 ** i - 1), random.randint(10 ** (i - 1) + 1, 10 ** i - 1))
         for i in range(3, 7)
         for _ in range(10*k)]

    random.shuffle(pairs)

    print("Multiplication nxm:", len(pairs))

    data = []
    for num1, num2 in pairs:
        answer = num1 * num2

        if random.random() < 0.5:
            num1, num2 = num2, num1

        question = f"{num1} * {num2}"
        if num2 > num1:
            num1, num2 = num2, num1

        num_digits_1 = len(str(num1))
        num_digits_2 = len(str(num2))
        if num1 % (10 ** (num_digits_1 - 1)) == 0 or num2 % (10 ** (num_digits_2 - 1)) == 0:
            cot = question + " = " + str(answer)
        else:
            num2_digits = [int(d) for d in str(num2)]

            split_terms = [d * 10 ** (len(num2_digits) - i - 1) for i, d in enumerate(num2_digits) if d != 0]
            split = f"""{num1} * ({" + ".join(str(x) for x in split_terms)})"""
            expansion = " + ".join([f"{num1} * {x}" for x in split_terms])
            summation_terms = [num1 * x for x in split_terms]
            summation = " + ".join(str(x) for x in summation_terms)
            step = ""
            while summation_terms:
                first = summation_terms.pop(0)
                if not summation_terms:
                    output = first
                    break
                summation_terms[0] = first + summation_terms[0]
                step = step + " + ".join([f"{x}" for x in summation_terms])
                if len(summation_terms) >= 2:
                    step = step + " = "

            cot = question + " = " + f"{split} = {expansion} = {summation} = " + step

        assert (cot.split()[-1] == str(answer))
        data.append({"input": question, "output": cot, "answer": str(answer)})
    return data


def div_n_1(k=1000):
    # Division n/1, with n up to 16 digits
    # pairs represent (divisor, quotient)
    pairs = \
        [(random.randint(2, 9), random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(1, 5)
         for _ in range(4*k)] + \
        [(random.randint(2, 9), random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(5, 8)
         for _ in range(8*k)] + \
        [(random.randint(2, 9), random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(8, 12)
         for _ in range(12*k)] + \
        [(random.randint(2, 9), random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(12, 17)
         for _ in range(16*k)] + \
        [(1, random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(1, 17)
         for _ in range(k//2)] + \
        [(10, random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(1, 17)
         for _ in range(k//2)] + \
        [(random.randint(10 ** (j - 1) + 1, 10 ** j), 0)
         for j in range(1, 17)
         for _ in range(k//10)] + \
        [(random.randint(1, 10), random.randint(1, 10))
         for _ in range(k//2)] + \
        [(0, random.randint(10 ** (j - 1) + 1, 10 ** j))
         for j in range(1, 18)
         for _ in range(k//10)]

    random.shuffle(pairs)

    print("Division n/1:", len(pairs))
    data = []
    for num1, num2 in pairs:
        # make it divisible with 0.5 probability
        if num1 > 1 and random.random() < 0.5:
            remainder = random.randint(1, num1 - 1)
        else:
            remainder = 0

        # divided by 0
        if num1 == 0:
            question = f"{num2} / {num1}"
            cot = question + " is " + "undefined"
            answer = "undefined"
            data.append({"input": question, "output": cot, "answer": answer})
            continue

        dividend = num1 * num2 + remainder

        question = f"{dividend} / {num1}"
        cot = question + " = " + str(num2) + " R " + str(remainder) if remainder != 0 else question + " = " + str(num2)
        answer = str(num2) + " R " + str(remainder) if remainder != 0 else str(num2)

        assert (cot.split()[-1] == answer.split()[-1])
        data.append({"input": question, "output": cot, "answer": answer})
    return data


def div_n_m(k=1000):
    # Division n/m with dividend<=12 digits and quotient<=7 digits
    # pairs represent (dividend, divisor)

    pairs = \
        [(random.randint(10 ** (j - 1) + 1, 10 ** j),
          random.randint(10 ** (i - 1) + 1, 10 ** i))
         for i in range(2, 7)
         for j in range(i + 1, i + 7)
         for _ in range(10*k)] + \
        [(random.randint(10 ** (j - 1) + 1, 10 ** j),
          random.randint(10 ** (i - 1) + 1, 10 ** i))
         for i in range(2, 7)
         for j in range(2, i + 7)
         for _ in range(k)]

    random.shuffle(pairs)

    print("Division n/m:", len(pairs))

    data = []
    for num1, num2 in pairs:

        quotient = num1 // num2
        remainder = num1 % num2

        # make it divisible with 0.5 probability
        if num1 > num2 and random.random() < 0.5:
            num1 = num1 - remainder
            quotient = num1 // num2
            remainder = num1 % num2

        question = f"{num1} / {num2}"

        if quotient == 0:
            cot = f"{num1} / {num2} = {quotient} R {remainder}"
            answer = f"{quotient} R {remainder}"
        elif num1 == num2:
            cot = f"{num1} / {num2} = {quotient}"
            answer = f"{quotient}"
        else:
            cot = ""
            left = num1

            i = 0
            computed_q = 0
            while left >= num2:
                if int(str(quotient)[i]) != 0:
                    intermediate = int(str(quotient)[i] + "0" * (len(str(quotient)) - 1 - i))
                    answer = num2 * intermediate
                    new_left = left - answer
                    step = f"{left} - {num2} * {intermediate} = {left} - {answer} = {new_left}\n"
                    cot = cot + step
                    left = new_left
                    computed_q = computed_q + intermediate
                i = i + 1

            assert (left == remainder)
            assert (computed_q == quotient)
            if remainder != 0:
                cot = cot + f"Therefore, {num1} / {num2} = {quotient} R {remainder}"
                answer = f"{quotient} R {remainder}"
            else:
                cot = cot + f"Therefore, {num1} / {num2} = {quotient}"
                answer = f"{quotient}"

        assert (cot.split()[-1] == answer.split()[-1])
        data.append({"input": question, "output": cot, "answer": answer})
    return data


def add_instructions(data, template_name='./templates.json'):
    # Add natural language instructions to the generated arithmetic data using template
    with open(template_name) as fp:
        template = json.load(fp)

    data_converted = []
    for instance in data:

        arithmetic = instance["input"]
        output_dict = {}

        # add noise to instruction
        if random.random() < 0.05:
            if " + " in arithmetic:
                arithmetic = "the sum of " + arithmetic.replace("+", "and")
            if " - " in arithmetic:
                arithmetic = "the difference of " + arithmetic.replace("-", "and")
            if " * " in arithmetic:
                arithmetic = "the product of " + arithmetic.replace("*", "and")
            if " / " in arithmetic:
                arithmetic = "the quotient and remainder of " + arithmetic.replace("/", "and")

        if random.random() < 0.5:
            arithmetic = arithmetic.replace("*", "x")

        if random.random() < 0.1:
            arithmetic = arithmetic.replace("+", "plus").replace("-", "minus")
            arithmetic = arithmetic.replace(" x ", " times ").replace("*", "multiplied by").replace("/", "divided by")

        if random.random() < 0.5:
            if "+" in arithmetic or "-" in arithmetic or "*" in arithmetic or "/" in arithmetic or "x" in arithmetic:
                arithmetic = arithmetic.replace(" ", "")

        num = random.randint(1, 50)

        instruction = template[str(num)].format(
            input=arithmetic
        )

        output_dict["instruction"] = instruction
        output_dict["input"] = instance["input"]
        output_dict["output"] = instance["output"]
        output_dict["answer"] = instance["answer"]

        data_converted.append(output_dict)

    return data_converted


def reformat(data):
    for i in range(len(data)):
        example = data[i]
        data[i] = {
            'text': example['instruction'] + ' ' + example['output']
        }
    return data


def make_splits(data, validation_frac=0.005):
    random.shuffle(data)

    validation_size = int(len(data)*validation_frac)
    test_size = validation_size
    train_size = len(data) - (validation_size + test_size)

    splits = {
        'train': data[:train_size],
        'valid': data[train_size:(train_size + validation_size)],
        'test': data[(train_size + validation_size):],
    }
    for k, v in splits.items():
        print(k, len(v), sep='\t')
    return splits


def save(splits, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        output_filename = os.path.join(output_dir, 'goat_%s.jsonl' % name)
        with open(output_filename, "w") as f:
            for item in split:
                f.write(json.dumps(item))
                f.write('\n')


def save_metadata(metadata, meta_dir):
    Path(meta_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(meta_dir, 'goat_metadata.json'), 'w') as f:
        json.dump(metadata, f)


def num_tokens(data):
    print("Counting tokens..")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    batch_size = 10000
    num_batches = len(data)//batch_size
    tokens = 0
    for i in trange(num_batches+1):
        batch = data[i*batch_size:(i+1)*batch_size]
        if len(batch) > 0:
            tokens_ = sum([
                len(x) for x in tokenizer.encode_batch(
                    [x['text'] for x in batch], disallowed_special=()
                )
            ])
            tokens += tokens_
    return tokens


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--templates-file', type=str, default="templates.json")
    parser.add_argument('--output-dir', type=str, default="data_jsonl")
    parser.add_argument('--meta-dir', type=str, default="meta_jsonl")
    parser.add_argument('--k', type=int, default=1000, help='controls number of samples per pair type')
    args = parser.parse_args()

    random.seed(123321)

    # Generate arithmetic data
    k = args.k
    data = addition(k) + subtraction(k) + mul_1_n(k) + mul_n_m(k) + div_n_1(k) + div_n_m(k)
    print("Arithmetic dataset generated")
    print("Total:", len(data))

    # Add instructions
    data = add_instructions(data, args.templates_file)
    print("Instructions added!")
    print("Total:", len(data))

    data = reformat(data)
    tokens = num_tokens(data)
    print("Tokens: %d" % tokens)

    splits = make_splits(data, validation_frac=0.005)

    save(splits, args.output_dir)
    save_metadata({
        'num_tokens': tokens,
        'num_examples': len(data)
    }, args.meta_dir)

    print("Done")
