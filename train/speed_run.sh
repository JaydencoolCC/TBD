uid="$(date +%Y%m%d_%H%M%S)"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 
micro_batch_size=1 
gradient_accumulation_steps=1
max_steps=-1
gpu_count=8

accelerate launch \
  --num_processes=${gpu_count} \
  train/sft.py \
  --deepspeed="train/deepspeed.json" \
  --block_size=16384 \
  --per_device_train_batch_size=${micro_batch_size} \
  --per_device_eval_batch_size=${micro_batch_size} \
  --gradient_accumulation_steps=${gradient_accumulation_steps} \
  --num_train_epochs=${epochs} \
  --warmup_ratio=0.05 \
  --bf16=True \
  --eval_strategy=no \
  --logging_steps=1 \
  --save_strategy=no \
  --lr_scheduler_type=cosine \
  --learning_rate=${lr} \
  --weight_decay=${weight_decay} \
  --adam_beta1=0.9 \
  --adam_beta2=0.95 \
  --output_dir="ckpt/${uid}" \
  --save_only_model=True \
  --gradient_checkpointing=True \
  --accelerator_config="{\"gradient_accumulation_kwargs\": {\"sync_each_batch\": true}}"