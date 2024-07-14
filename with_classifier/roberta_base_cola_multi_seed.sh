# export num_gpus=1
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./cola"
export CUDA_VISIBLE_DEVICES=0

for seed in 0 1 2 3 4 
do
python \
run_glue_no_trainer_save_best.py \
--task_name cola \
--max_length 512 \
--pad_to_max_length \
--model_name_or_path /home/cver4090/Project/Pretrained/RoBERTa/base \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 8 \
--learning_rate 1e-2 \
--learning_rate_head 4e-3 \
--weight_decay 0.1 \
--num_train_epochs 30 \
--output_dir $output_dir/model \
--seed $seed \
--warmup_ratio 0.06 \
--rank 1024 \
--per_save_freq 3


python \
test_glue_no_trainer.py \
--task_name cola \
--max_length 512 \
--pad_to_max_length \
--model_name_or_path $output_dir/model/$seed/latest \
--per_device_eval_batch_size 8 \
--output_dir $output_dir/logs \
--seed $seed \
--rank 1024


done

