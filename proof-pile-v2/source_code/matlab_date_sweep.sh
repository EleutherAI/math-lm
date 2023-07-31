#!/bin/bash

for YEAR in {2009..2022}
do 
    echo "YEAR IS ${YEAR}"
    python process_github.py --limit 100000 --batch-by-week --langs matlab --data-dir matlab_by_year_alt/data_jsonl${YEAR} --meta-dir matlab_by_year_alt/meta_json${YEAR} --repos-dir matlab_by_year_alt_repos/repos${YEAR} --init-date ${YEAR}-01-01 --search-end-date $((YEAR+1))-01-01
done 
