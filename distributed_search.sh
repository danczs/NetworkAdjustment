#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC channel_search_distributed.py "$@"