
### <- set paths - >

DATA_PATH="../../data/input_to_DNNs"
declare -a DATA_DIR=("$DATA_PATH/MRDA/" "$DATA_PATH/SWBD/")

CNN_SCR="dnn_scripts/cnn_aidr.py"
EXP_DIR="experiments/cnn_results/"
MODEL_DIR="saved_models/cnn_models/"

mkdir -p $MODEL_DIR
mkdir -p $EXP_DIR

###<- Set general DNN settings ->
dr_ratios=(0.0 0.2 0.4) #dropout_ratio
mb_sizes=(16 32 64 128) #minibatch-size

### <- set CNN settings ->
nb_filters=(150 200 250 300) #no of feature map
filt_lengths=(2 3 4)
pool_lengths=(2 3 4)

vocab_sizes=(80 85 90 95) # how many words in percentage for vocabulary

init_type="random" #instead of word embedding?

for data in "${DATA_DIR[@]}"; do
	d_set=$(echo $data | rev | cut -d '/' -f 2 | rev);
	log="$EXP_DIR/CNN_$d_set.log";
	echo "INFORMATION: CNN on $d_set" >> "$log";

	for ratio in ${dr_ratios[@]}; do
		for mb in ${mb_sizes[@]}; do
			for nb_filter in ${nb_filters[@]}; do
				for filt_len in ${filt_lengths[@]}; do
					for pool_len in ${pool_lengths[@]}; do
						for vocab in ${vocab_sizes[@]}; do

							echo "INFORMATION: dropout_ratio=$ratio minibatch-size=$mb filter-nb=$nb_filter filt_len=$filt_len pool_len=$pool_len vocab=$vocab" >> $log;
							echo "----------------------------------------------------------------------" >> $log;

							python $CNN_SCR --data-dir=$data --model-dir=$MODEL_DIR --init-type=$init_type\
							--vocabulary-size=$vocab --dropout_ratio=$ratio --minibatch-size=$mb\
							--nb_filter=$nb_filter --filter_length=$filt_len --pool_length=$pool_len\
							--vocabulary-size=$vocab -D $d_set >>$log
							wait

							echo "----------------------------------------------------------------------" >> $log;

						done
					done
				done
			done 
		done	
	done 
done
