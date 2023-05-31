#!/bin/bash

mkdir -p raw_pilev2/arXiv

# not sure what the credentials you need, but it just works on stability cluster
aws s3 cp s3://s-eai-neox/data/pilev2/pilev2_local_deduped/arXiv_ver2/ raw_pilev2/arXiv --recursive


