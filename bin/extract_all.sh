#!/usr/bin/env bash
# conda activate vissl
bin/extract_features.sh simclr_dir
bin/extract_features.sh swav_dir
bin/extract_features.sh cntr_dir
bin/extract_features.sh slfdstl_dir
bin/extract_features.sh simplecntr_dir
bin/extract_features.sh simplecntr128_dir