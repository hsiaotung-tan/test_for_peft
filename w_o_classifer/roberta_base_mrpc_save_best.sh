# export num_gpus=1
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./mrpc"
export CUDA_VISIBLE_DEVICES=0
python \
run_glue_no_trainer_save_best.py \
--task_name mrpc \
--max_length 512 \
--pad_to_max_length \
--model_name_or_path /home/tanxiaodong/Project/LLM/Pretrained/RoBERTa/base \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 8 \
--learning_rate 1e-2 \
--weight_decay 0.1 \
--num_train_epochs 30 \
--output_dir $output_dir/model \
--seed 0 \
--warmup_ratio 0.06 \
--rank 1024 \
--per_save_freq 3