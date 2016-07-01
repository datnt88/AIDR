
### <- set paths - >

DATA_PATH="../data/nepal"
declare -a DATA_DIR=("$DATA_PATH/1/" "$DATA_PATH/2/" "$DATA_PATH/3/" "$DATA_PATH/4/" "$DATA_PATH/5/")

MLP_SCR="MLP_aidr.py"
EXP_DIR="../experiments/MLP_results/"
MODEL_DIR="../saved_models/MLP_models/"

mkdir -p $MODEL_DIR
mkdir -p $EXP_DIR

###<- Set general DNN settings ->
dr_ratios=(0.0 0.2 0.3 0.4 0.5) #dropout_ratio
mb_sizes=(16 32) #minibatch-size

h_sizes=(50 100 150 200)
nb_layers=(0 1 2)


for data in "${DATA_DIR[@]}"; do
	for nb_layer in ${nb_layers[@]}; do
		d_set=$(echo $data | rev | cut -d '/' -f 2 | rev);
		log="$EXP_DIR/MLP-layers.$nb_layer.$d_set.log";
		>$log
		
		echo "INFORMATION: MLP with layer $nb_layer on $d_set ta" >> "$log";

		for mb in ${mb_sizes[@]}; do
			for dr in ${dr_ratios[@]}; do
				for h_size in ${h_sizes[@]}; do

						echo "INFORMATION: dropout_ratio=$dr minibatch-size=$mb hidden-zie=$h_size layer-nb=$nb_layer" >> $log;
						echo "----------------------------------------------------------------------" >> $log;

						cmd="$MLP_SCR --data-dir=$data --model-dir=$MODEL_DIR --dropout_ratio=$dr -F 0\
						--minibatch-size=$mb --nb-layers=$nb_layer -C 1 --hidden-size=$h_size -D $d_set"
						
#						echo $cmd >>$log
						echo $cmd
						exit 
						
#						python  $cmd >>$log
						
						wait

						echo "----------------------------------------------------------------------" >> $log;

				done
			done
		done
	done 
done	
