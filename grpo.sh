gpu_ids=0,1,2,3
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)

export CUDA_VISIBLE_DEVICES=$gpu_ids
export HF_ENDPOINT="https://hf-mirror.com"
export DECORD_EOF_RETRY_MAX=20480
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

MODEL_PATH=output/Qwen2.5-VL-7B-Instruct-sft
RUN_NAME=Qwen2.5-VL-7B-Instruct-grpo
SAVE_PATH=output/${RUN_NAME}

rm ${SAVE_PATH}/*

python3 -m verl.trainer.main \
    config=scripts/grpo_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${n_gpu} \
    trainer.save_checkpoint_path=${SAVE_PATH} \

python3 model_merger.py --local_dir ${SAVE_PATH}
rm ${SAVE_PATH}/*.pt