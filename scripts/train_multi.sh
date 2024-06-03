export CLS_TOKEN1="teddybear"
export CLS_TOKEN2="barn"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="ckpt/${CLS_TOKEN1}_${CLS_TOKEN2}"

accelerate launch train_class_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list=config/concept.json \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>+<new2>" \
  --use_spl \
  --spl_weight=1.0 \
  --cls_token "${CLS_TOKEN1}+${CLS_TOKEN2}" \
  --no_safe_serialization \
  --report_to "wandb" \
  --validation_steps=25 \
  --validation_prompt="a <new1> ${CLS_TOKEN1} sitting in front of a <new2> ${CLS_TOKEN2}" \