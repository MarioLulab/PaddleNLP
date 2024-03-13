export FLAGS_call_stack_level=3
export NVIDIA_TF32_OVERRIDE=0

export CUDA_VISIBILE_DEVICES=4,5,6,7

task_name="llama_auto_dy2st_bs8_fp16_dp2mp2pp2"
case_out_dir="output/$task_name"
case_log_dir="output/$task_name""_log"
rm -rf $case_out_dir
rm -rf $case_log_dir

python -u -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir $case_log_dir run_pretrain_auto.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "/luqi13/paddle-docker/PaddleNLP/llm/data" \
    --output_dir $case_out_dir \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --hidden_size 1024 \
    --intermediate_size 3072 \
    --num_hidden_layers 8 \
    --num_attention_heads 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --fp16 0 \
    --fp16_opt_level "O2" \
    --amp_master_grad 0 \
    --amp_custom_white_list "lookup_table" "lookup_table_v2" "fused_rms_norm" \
    --amp_custom_black_list "softmax_with_cross_entropy" \
    --scale_loss 1024 \
    --pipeline_parallel_degree 2 \
    --tensor_parallel_degree 2 \
    --sharding_parallel_degree 1 \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --max_steps 10 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --sharding "stage1" \
    --eval_steps 1000000 \
    --disable_tqdm true \
    --continue_training 0 \
    --recompute 0 \
    --recompute_use_reentrant true \
    --recompute_granularity "full" \
    --use_flash_attention 0 \
    --fuse_attention_qkv 0 \
    --fuse_attention_ffn 0 \
    --use_fused_rope 1 \
    --use_fused_rms_norm 0 \
    --do_train \
    --do_eval \
    --device "gpu" \
    --data_impl "mmap" \
    --enable_auto_parallel 1 \
    --to_static 1 \
    --max_grad_norm 1.0 \
    # --num_hidden_layers 4 \