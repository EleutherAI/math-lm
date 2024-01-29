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
    --inputpath $FILTERED_DATA \
    --destpath $SAVE_DESTINATION \
    --judgement_file $PATH_TO_SAVED_JUDGEMENTS
```
The following are criteria for the answers of accepted examples.
```
1. Answers should directly address the question. There should be minimal tangents and digressions.
2. Answers should be written in high quality English prose.
3. Answers should be valid markdown. 
4. Answers should not reference other answers or comments (e.g to elaborate on @user3413's answer...), and should not include personal details about the author (e.g "I published a paper that shows the solution is...").
5. Answers should be complete, e.g "the rest of the argument is an exercise left to the reader" is disallowed.
6. Answers should not contain URLs. References to well-known papers and textbooks are fine.
7. If information from the question required to deduce the answer was lost or corrupted (e.g the question references a missing image, and that image is necessary to answer the question), exclude that answer.
```
Note that since we are only supervising on the answer and not the question, having some poor quality questions is fine, and long as the answer is a strong response to the question.
