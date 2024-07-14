# export num_gpus=1
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./mrpc"
export CUDA_VISIBLE_DEVICES=0
python \
test_glue_no_trainer.py \
--task_name mrpc \
--max_length 512 \
--pad_to_max_length \
--model_name_or_path /home/tanxiaodong/Project/LLM/VeRA/current/mrpc/model/best \
--per_device_eval_batch_size 8 \
--output_dir $output_dir/logs \
--seed 0 \
--rank 1024

