# Instruction Tuning Data

This directory contains scripts for creating an instruction tuning dataset made up of highly-rated stack exchange answers. This process is composed of three steps:
1. Download and preprocess stack exchange answers from the internet archive XML dump.
2. Filter for highly rated answers.
3. Use the data filtering UI to manually filter out bad examples.

### Step 1
Run
```
python get_stackexchange.py
```
This takes over an hour, and this time is dominated by downloading math stack exchange. If you just want some small amount of data for testing, comment out every `get_and_format` call except one to a small stack exchange, such as CS Theory.

By default, datasets are saved to `stack-exchange/${NAME_OF_STACK_EXCHANGE}/unfiltered.jsonl`.

### Step 2
Run
```
python filter_for_score.py \
    --inputpath $UNFILTERED_DATA \
    --destpath filtered-stack-exchange/$SAVE_DESTINATION \
    --min_answer_votes $MIN_NUM_VOTES
```

### Step 3
Open the notebook `data_viewer.ipynb`. It contains an example of how to use the `nbviewer`-based data filtering UI. The UI will show you one QA example at a time, which you can accept or reject. Acceptances and rejections made in the UI are synced with a yaml file. To filter the data jsonl to just those accepted by the human using the UI, run
```
python filter_by_human.py \
    --inputpath $FILTERE_DATA \
    --destpath $SAVE_DESTINATION \
    --judgement_file $PATH_TO_SAVED_JUDGEMENTS
```
