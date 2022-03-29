#!/usr/bin/env bash
# conda activate vissl
bin/extract_features.sh simclr_dir
bin/extract_features.sh swav_dir
bin/extract_features.sh cntr_dir
bin/extract_features.sh dstl_dir
bin/extract_features.sh dstlema_dir
bin/extract_features.sh dstlrank_dir
bin/extract_features.sh simplecntr_dir
bin/extract_features.sh simplecntr128_dir
bin/extract_features.sh simplecntr512_dir
bin/extract_features.sh cntr128_dir


bin/extract_features.sh dstlasym_dir
bin/extract_features.sh dstlzdim_dir
bin/extract_features.sh dstllong_dir