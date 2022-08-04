#!/usr/bin/env bash

for DATA in flowers pets caltech cars aircrafts dtd food101 sun397
do
    bin/trsnfeval_nlprun.sh dissl_zdim8_long_dir _z8192 $DATA
    bin/trsnfeval_nlprun.sh swav_queuelong_dir "" $DATA
    bin/trsnfeval_nlprun.sh simclr_nomulti_long_dir "" $DATA
done
