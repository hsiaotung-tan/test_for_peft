# export num_gpus=8
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./cola"
python \
run_glue.py \
--model_name_or_path /home/cver4090/Project/Pretrained/RoBERTa/base \
--task_name cola \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 512 \
--per_device_train_batch_size 64 \
--learning_rate 3e-4 \
--num_train_epochs 80 \
--output_dir $output_dir/model \
--logging_steps 10 \
--logging_dir $output_dir/log \
--eval_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--seed 0 \
--weight_decay 0.1 \
--rank 1024