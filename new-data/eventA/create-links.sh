#!/bin/bash
num_of_folds=5
orig_filename=nepal_eq_comb_4exp_header.csv
data=/alt/work/kmannai/AIDR-DA-ALT-SC/dat-data/${num_of_folds}-folds


for i in 1 2 3 4 5
	do

	#create dir
	mkdir $i

	#create softlinks to:
	#1. train
	ln -s $data/${orig_filename%.*}-train-chunk${i}.csv $i/train.csv

	#test
	ln -s $data/${orig_filename%.*}-test-chunk${i}.csv $i/test.csv

	#dev
	ln -s $data/${orig_filename%.*}-dev.csv $i/dev.csv

	done

