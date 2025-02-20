export PYTHONPATH=$PWD:$PYTHONPAT
gpuid="0,1,2,3"
master_port=10012

num_encoder_layers=2
num_decoder_layers=4
seed=333
model=ras
learning_rate=0.0007

config="./config/mvtec/config_mvtec.yaml"
model_name="experiments/RAS_mvtec/"${model}"_numberlayer-("${num_encoder_layers}","${num_decoder_layers}")_lr-"${learning_rate}"_seed_"${seed}

# learning_rate=0.001
# config="./config/visa/config_visa.yaml"
# model_name="experiments/RAS_visa/"${model}"_numberlayer-("${num_encoder_layers}","${num_decoder_layers}")_lr-"${learning_rate}"_seed_"${seed}

# config="./config/mpdd/config_mpdd.yaml"
# model_name="experiments/RAS_mpdd/"${model}"_numberlayer-("${num_encoder_layers}","${num_decoder_layers}")_lr-"${learning_rate}"_seed_"${seed}

# learning_rate=0.0009
# config="./config/btad/config_btad.yaml"
# model_name="experiments/RAS_btad/"${model}"_numberlayer-("${num_encoder_layers}","${num_decoder_layers}")_lr-"${learning_rate}"_seed_"${seed}


CUDA_VISIBLE_DEVICES=${gpuid} python -m torch.distributed.launch --nproc_per_node=4 --master_port=${master_port} train_val.py --model_name ${model_name} --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --learning_rate ${learning_rate} --seed ${seed} --model ${model} --config ${config}