#!/usr/bin/env bash

for DATA in flowers pets caltech cars aircrafts dtd food101  cifar10 cifar100 #sun397
do
    bin/trsnfeval_sklearn_nlprun.sh dissl_e400_d8192_m6_dir _z8192 $DATA
    #bin/trsnfeval_sklearn_nlprun.sh swav_queuelong_dir "" $DATA
    #bin/trsnfeval_sklearn_nlprun.sh simclr_nomulti_long_dir "" $DATA

    #python bin/collect_results_linear.py dissl_zdim8_long_dir/trnsf/$DATA/
    #python bin/collect_results_linear.py swav_queuelong_dir/trnsf/$DATA/
    #python bin/collect_results_linear.py simclr_nomulti_long_dir/trnsf/$DATA/
done
