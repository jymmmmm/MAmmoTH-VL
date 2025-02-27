export HF_HOME="/data/yiming/data/huggingface_cache"
export MLP_WORKER_0_PORT=2591 
FINAL_RUN_NAME="/data/yiming/data/v4_checkpoints"
Task_Name="VisualWebInstruct_mammoth_vl"
LOG_DIR="/data/yiming/data/eval_logs/eval6"



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks  mmmu_pro_vision\
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${Task_Name} \
    --output_path ${LOG_DIR} \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks  mathvista_testmini\
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${Task_Name} \
    --output_path ${LOG_DIR} \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks  mmvet\
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${Task_Name} \
    --output_path ${LOG_DIR} \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks  mathverse_testmini_vision\
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${Task_Name} \
    --output_path ${LOG_DIR} \



# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks mmvet --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path ${LOG_DIR}&
# CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks ai2d  --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path ${LOG_DIR} &
# CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks mmstar --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path ${LOG_DIR} &
# CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks chartqa --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path ${LOG_DIR} &
# CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks mmvet  --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path ${LOG_DIR} &
# CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks llava_wilder_small  --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path ${LOG_DIR} &
# wait
