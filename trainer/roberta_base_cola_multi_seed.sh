# export num_gpus=8
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export output_dir="./cola"
for seed in 0 1 2 3 4
do
python \
run_glue.py \
--model_name_or_path /home/cver4090/Project/Pretrained/RoBERTa/base \
--task_name cola \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 512 \
--per_device_train_batch_size 64 \
--learning_rate 1e-2 \
--lr_scheduler_type linear \
--warmup_ratio 0.06 \
--num_train_epochs 80 \
--output_dir $output_dir/model/$seed \
--logging_steps 500 \
--logging_dir $output_dir/log/$seed \
--eval_strategy epoch \
--save_strategy steps \
--save_steps 0.25 \
--seed $seed \
--weight_decay 0.1 \
--rank 1024
done
