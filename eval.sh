gpu_ids=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpu_ids
export DECORD_EOF_RETRY_MAX=20480

MODEL_NAME_OR_PATH=Qwen2.5-VL-7B-Instruct-grpo
BATCH_SIZE=32
TOKENS_PER_IMAGE=256
MAX_FRAMES=32
TASK=vsibench,videommmu,mmvu,mvbench,tempcompass,videomme

python eval.py --model_path output/$MODEL_NAME_OR_PATH --task $TASK \
 --batch_size $BATCH_SIZE --max_frames $MAX_FRAMES --max_pixels $((TOKENS_PER_IMAGE*28*28)) \
 --name $MODEL_NAME_OR_PATH-$MAX_FRAMES