#!/bin/bash

# Name of the file containing URLs
file="urls.txt"

# Name of the directories to download files to
dir_train="data_jsonl/train"
dir_val="data_jsonl/validation"
dir_test="data_jsonl/test"

# Check if the directories exist, create them if they don't
mkdir -p $dir_train $dir_val $dir_test

# Initialize a counter
counter=0

# Read the file line by line
while IFS= read -r url
do
    # Generate a filename using the counter
    filename=$(printf "arXiv_%03d.jsonl" $counter)
    tempfile=$(printf "temp_%03d.jsonl" $counter)

    # Download the file with a progress bar
    echo "Downloading $url"
    wget --progress=bar:force $url -O $tempfile 2>&1
    echo "Saving splits..."
    
    # Calculate the number of lines for each set
    total_lines=$(wc -l < $tempfile)
    train_lines=$(($total_lines * 99 / 100))
    val_test_lines=$(($total_lines - $train_lines))
    val_lines=$(($val_test_lines / 2))
    test_lines=$(($val_test_lines - $val_lines))
    
    # Save the lines to the respective files
    head -n $train_lines $tempfile > $dir_train/$filename
    tail -n $val_test_lines $tempfile | head -n $val_lines > $dir_val/$filename
    tail -n $test_lines $tempfile > $dir_test/$filename
    
    # Remove the temporary file
    rm $tempfile

    # Increment the counter
    counter=$((counter+1))
done < "$file"
