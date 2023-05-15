#!/bin/bash

mkdir -p raw_pilev2/GithubDiff_ver2
mkdir -p raw_pilev2/GithubIssues

# not sure what the credentials you need, but it just works on stability cluster

aws s3 cp s3://s-eai-neox/data/pilev2/pilev2_local_deduped/GithubDiff_ver2/ raw_pilev2/GithubDiff --recursive

aws s3 cp s3://s-eai-neox/data/pilev2/pilev2_local_deduped/GithubIssues/ raw_pilev2/GithubIssues --recursive

