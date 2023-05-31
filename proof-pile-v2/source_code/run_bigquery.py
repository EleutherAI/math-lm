"""
To setup bigquery, see https://cloud.google.com/python/docs/reference/bigquery/latest

NOTE: running (large) queries costs money.
"""
from google.cloud import bigquery
import json
import gzip
import shutil
import os


def run_query(query):
    client = bigquery.Client()
    query_job = client.query(query)
    rows = query_job.result()
    return rows


def _to_gz(f_jsonl):
    with open(f_jsonl, 'rb') as f_in:
        with gzip.open(f_jsonl+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(f_jsonl)


def save_rows(rows, outdir, lang, shard_size):
    shard = 0
    for i, row in enumerate(rows):
        f_jsonl = '%s/original/%s/math-lm-%s-%d.jsonl' % (outdir, lang, lang, shard)
        with open(f_jsonl, 'a') as f:
            f.write(json.dumps(dict(row)))
            f.write('\n')
        if (i+1) % shard_size == 0:
            shard += 1
            _to_gz(f_jsonl)
    _to_gz(f_jsonl)


def main(args):
    query = open(args.query, 'r').read()
    rows = run_query(query)
    save_rows(rows, args.outdir, args.lang, args.shard_size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='bigquery-code/original/coq/query.sql')
    parser.add_argument('--outdir', type=str, default='bigquery-code')
    parser.add_argument('--lang', type=str, default='coq')
    parser.add_argument('--shard-size', type=int, default=10000)

    args = parser.parse_args()
    main(args)
