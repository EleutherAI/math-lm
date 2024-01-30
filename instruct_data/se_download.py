from dataclasses import dataclass, field, fields
from functools import lru_cache
from xml.etree import ElementTree
from datetime import datetime
from enum import Enum
import typing
from typing import List, Optional, Union
import os.path
from itertools import groupby
import dataclasses
from tqdm import tqdm
from bs4 import BeautifulSoup
import sys
from pathlib import Path
import tarfile
import random
import ndjson
import json
"""
Author: E.W.Ayers

This code takes a dump of math overflow XML and produces
a structured set of questions with answers.

1. Get mathoverflow.net.7z file
2. Extract this to `DATA_DIR = 'data/mathoverflow.net'`
3. Run `questions()` and run it to get a dictionary of mathoverflow questions.
   Each question has an `Answers` field that contains a list of answers for the given q.
"""


def batch_loader(seq, size):
    """
    Iterator that takes in a list `seq` and returns
    chunks of size `size`
    """
    return [seq[pos : pos + size] for pos in range(0, len(seq), size)]


DOC_SEP = "<|endoftext|>"

# source: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
class PostType(Enum):
    Question = 1
    Answer = 2
    OrphanedTagWiki = 3
    TagWikiExcerpt = 4
    TagWiki = 5
    ModeratorNomination = 6
    WikiPlaceholder = 7
    PrivilegeWiki = 8


def is_optional(field):
    return typing.get_origin(field) is Union and type(None) in typing.get_args(field)


def fromXML(cls, element):
    out = {}
    for field in fields(cls):
        field_key = field.name
        field_type = field.type
        f = field.metadata.get("from_xml")
        if f == "skip":
            continue
        attr_key = f["key"] if (f is not None and f["key"] is not None) else field_key
        v = element.attrib.get(attr_key)
        if v is None:
            if field.default is not dataclasses.MISSING:
                out[field_key] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                out[field_key] = field.default_factory()  # type: ignore
            elif is_optional(field_type):
                out[field_key] = None
            else:
                raise Exception(f"Missing field {attr_key}")
            continue
        if is_optional(field_type):
            field_type = typing.get_args(field_type)[0]
        if f is not None and f["fn"] is not None:
            out[field_key] = f["fn"](v)
        elif field_type is int:
            out[field_key] = int(v)
        elif field_type is str:
            out[field_key] = str(v)
        elif field_type is datetime:
            out[field_key] = datetime.fromisoformat(v)
        else:
            raise Exception(f"Don't know how to decode {field_type}")
    return cls(**out)


def use(fn, key=None):
    return field(metadata={"from_xml": {"fn": fn, "key": key}})


def skip(default):
    return field(default=default, metadata={"from_xml": "skip"})


def iter_rows(path):
    for [_, element] in ElementTree.iterparse(path, events=["start"]):
        if element.tag == "row":
            yield element


DATA_DIR = "nothing"


@dataclass
class Comment:
    Id: int
    PostId: int
    Score: int
    Text: str
    CreationDate: datetime
    UserId: Optional[int]


# lru_cache()
def comments():
    path = os.path.join(DATA_DIR, "Comments.xml")
    out = {}
    for element in iter_rows(path):
        x: Comment = fromXML(Comment, element)
        out[x.Id] = x
    print(f"Processed {len(out)} comments.")
    return out


@dataclass
class Post:
    Id: int
    CreationDate: datetime
    DeletionDate: Optional[datetime]
    Score: int
    Body: str  # in html; need to parse out?
    Title: Optional[str]
    OwnerUserId: Optional[int]
    ViewCount: Optional[int]
    AcceptedAnswerId: Optional[int]
    ParentId: Optional[int]
    PostType: "PostType" = use(lambda x: PostType(int(x)), "PostTypeId")
    Comments: List[Comment] = skip(None)
    Answers: Optional[List["Post"]] = skip(None)
    Tags: str = field(default="")


# @lru_cache()
def questions():
    path = os.path.join(DATA_DIR, "Posts.xml")
    cs = {}
    for k, c in groupby(comments().values(), lambda c: c.PostId):
        x = list(c)
        x.sort(key=lambda x: -x.Score)
        cs[k] = x
    qs = {}
    answers = {}
    for element in iter_rows(path):
        post = fromXML(Post, element)
        post.Comments = cs.get(post.Id, [])
        if post.PostType is PostType.Question:
            post.Answers = []
            qs[post.Id] = post
        elif post.PostType is PostType.Answer:
            answers[post.Id] = post
    for qk, aa in groupby(answers.values(), lambda a: a.ParentId):
        x = list(aa)
        x.sort(key=lambda x: -x.Score)
        qs[qk].Answers = x
    print(f"Processed {len(qs)} questions with {len(answers)} answers.")
    return qs


def strip_html(string):
    soup = BeautifulSoup(string, "html.parser")
    return soup.get_text()


def row_of_post(post):
    input_text = strip_html(post.Body).strip()
    input_score = post.Score

    commented = False

    answered = False
    best_output_text = ""
    best_output_score = -1
    

    for answer in post.Answers:
        if answer.Score > best_output_score:
            answered = True
            best_output_score = answer.Score
            best_output_text = strip_html(answer.Body).strip()


    return {
            "input": input_text,
            "output": best_output_text,
            "meta": {
                "post_id": post.Id,
                "input_score": input_score,
                "output_score": best_output_score,
                "post_title": post.Title,
                }
            }


def get_and_format(url, save_dir):
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    archive_path = os.path.join(save_dir, "archive.7z")
    if not os.path.isfile(archive_path):
        os.system(f"wget -O {archive_path} {url}")

    global DATA_DIR
    DATA_DIR = os.path.join(save_dir, "xml")
    print(f"DATA DIR {DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        os.system(f"7z e {archive_path} -o{DATA_DIR}")

    print("parsing xml...")
    qs = questions()

    print("converting xml to text...")
    qs_dataset = [row_of_post(qs[key]) for key in tqdm(qs.keys())]

    shard_path = os.path.join(save_dir, "unfiltered.jsonl")
    with open(shard_path, "a+") as f:
        for row in tqdm(qs_dataset):
            f.write(json.dumps(row))
            f.write("\n")

    os.system(f"rm -r {DATA_DIR}")


if __name__ == "__main__":
    get_and_format(
        "https://archive.org/download/stackexchange/mathoverflow.net.7z",
        save_dir="stack-exchange/math_overflow",
    )
    get_and_format(
        "https://archive.org/download/stackexchange/math.stackexchange.com.7z",
        "stack-exchange/math_stack_exchange",
    )
    get_and_format(
        "https://archive.org/download/stackexchange/physics.stackexchange.com.7z", 
        "stack-exchange/physics_stack_exchange", 
    )
    get_and_format(
        "https://archive.org/download/stackexchange/cstheory.stackexchange.com.7z", 
        "stack-exchange/cstheory_stack_exchange", 
    )
    get_and_format(
        "https://archive.org/download/stackexchange/datascience.stackexchange.com.7z", 
        "stack-exchange/datascience_stack_exchange", 
    )
    get_and_format(
        "https://archive.org/download/stackexchange/proofassistants.stackexchange.com.7z", 
        "stack-exchange/proofassistants_stack_exchange",
    )
