
### <- set paths - >

DATA_PATH="../data/"
declare -a DATA_DIR=("$DATA_PATH/exp1/" )    # "$DATA_PATH/2/" "$DATA_PATH/3/" "$DATA_PATH/4/" "$DATA_PATH/5/")

CNN_SCR="CNN_with_feat.py"
EXP_DIR="../new_experiments/cnn_results/"
MODEL_DIR="../new_saved_models/cnn_models/"

mkdir -p $MODEL_DIR
mkdir -p $EXP_DIR

###<- Set general DNN settings ->
dr_ratios=(0.5) #dropout_ratio
mb_sizes=(64 128) #minibatch-size

### <- set CNN settings ->
nb_filters=(150) #no of feature map
filt_lengths=(2)
pool_lengths=(3 )
vocab_sizes=(85) # how many words in percentage for vocabulary


### <- embedding file ->
init_type="pretrained"
emb_file="../data/pretrained-vectors/GoogleNews-vectors-negative300.bin.txt.gz" 


#mlp related
#h_sizes=(50 100 150 200)
#nb_layers=(0 1 2)
nb_layers=(1)



for data in "${DATA_DIR[@]}"; do
	d_set=$(echo $data | rev | cut -d '/' -f 2 | rev);
	log="$EXP_DIR/CNN-with-feat-$d_set.log";
	>$log

	for nb_layer in ${nb_layers[@]}; do
		for ratio in ${dr_ratios[@]}; do
			for mb in ${mb_sizes[@]}; do
				for nb_filter in ${nb_filters[@]}; do
					for filt_len in ${filt_lengths[@]}; do
						for pool_len in ${pool_lengths[@]}; do
							for vocab in ${vocab_sizes[@]}; do


								CMD="$CNN_SCR --data-dir=$data --model-dir=$MODEL_DIR --init-type=$init_type\
								--vocabulary-size=$vocab --dropout_ratio=$ratio --minibatch-size=$mb\
								--nb_filter=$nb_filter --filter_length=$filt_len --pool_length=$pool_len\
								--nb-layers=$nb_layer -f $emb_file -D $d_set"
								
								echo "INFORMATION: $CMD" >> $log;
								echo "----------------------------------------------------------------------" >> $log;
								python $CMD >>$log
								echo $CMD

								echo "----------------------------------------------------------------------" >> $log;

							done
						done
					done
				done 
			done	
		done 
	done	
done
