export CLS_TOKEN="dog"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./ckpt/${CLS_TOKEN}_cls"
export INSTANCE_DIR="./data/${CLS_TOKEN}"

accelerate launch train_class_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a <new1> ${CLS_TOKEN}"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>" \
  --no_safe_serialization \
  --use_spl \
  --spl_weight=1 \
  --cls_token "${CLS_TOKEN}" \
  --report_to "wandb" \
  --validation_steps=50 \
  --validation_prompt="a <new1> ${CLS_TOKEN} is swimming" \
  --tracker_name "custom-diffusion-multi" \
  