export PYTHONHASHSEED=0
export output_dir="./mrpc"
export CUDA_VISIBLE_DEVICES=0
python \
test_glue.py \
--task_name mrpc \
--max_length 512 \
--pad_to_max_length \
--model_name_or_path /home/tanxiaodong/Project/LLM/VeRA/no_trainer/mrpc/model \
--per_device_eval_batch_size 8 \
--output_dir $output_dir/logs \
--seed 0 \
--rank 1024
