### <- set paths - >
#DATA_PATH="../../data/input_to_DNNs"
#declare -a DATA_DIR=("$DATA_PATH/MRDA/" "$DATA_PATH/SWBD/")

CNN_SCR="/Users/ndat/Desktop/aidr-final/CNN_MLP_Shafiq_May2/CNN_Fusion/cnn_aidr.py"
MODEL_DIR="IN_models/"

data=/Users/ndat/Desktop/aidr-final/weighted-adp/nepal/ev


#log=./mix-in-domain-embGG.log
log=./log.in_model



mkdir -p $MODEL_DIR


###<- Set general DNN settings ->
dr_ratios=(0.5) #dropout_ratio
mb_sizes=(128) #minibatch-size

### <- set CNN settings ->
nb_filters=(150) #no of feature map
filt_lengths=(2)
pool_lengths=(3) 

vocab_sizes=(90) # how many words in percentage for vocabulary

### <- embedding file ->
init_type="pretrained"
#emb_file="/home/local/QCRI/ndat/toolkits/DCNN/data/word2vec_source/trunk/NEW_crisis_vectors.skip.window6.count2.bin.text"
emb_file="/Users/ndat/Desktop/aidr-final/embeddings/NEW_crisis_vectors.skip.window6.count2.bin.text"

for ratio in ${dr_ratios[@]}; do
	for mb in ${mb_sizes[@]}; do
		for nb_filter in ${nb_filters[@]}; do
			for filt_len in ${filt_lengths[@]}; do
				for pool_len in ${pool_lengths[@]}; do
					for vocab in ${vocab_sizes[@]}; do
							echo "INFORMATION: dropout_ratio=$ratio minibatch-size=$mb filter-nb=$nb_filter filt_len=$filt_len pool_len=$pool_len vocab=$vocab" >> $log;
							echo "----------------------------------------------------------------------" >> $log;

							#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
                                                        python $CNN_SCR \
							--data-dir=$data/ --model-dir=$MODEL_DIR -i $init_type -f $emb_file\
							--vocabulary-size=$vocab --dropout_ratio=$ratio --minibatch-size=$mb\
							--nb_filter=$nb_filter --filter_length=$filt_len --pool_length=$pool_len\
							--vocabulary-size=$vocab  --data-spec="in"  >>$log
							wait

							echo "----------------------------------------------------------------------" >> $log;

					done
				done
			done
		done 
	done	
done 
