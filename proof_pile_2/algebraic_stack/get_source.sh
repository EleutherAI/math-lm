python process_stack.py -c $1
python process_github.py --langs isabelle matlab
python process_github.py --langs coq --limit 100000 --batch-by-week
python process_github.py --langs lean --limit 100000 --batch-by-week --init-date 2017-01-01
