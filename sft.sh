gpu_ids=0,1,2,3
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)

export CUDA_VISIBLE_DEVICES=$gpu_ids
export HF_ENDPOINT="https://hf-mirror.com"
export DECORD_EOF_RETRY_MAX=20480
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

llamafactory-cli train scripts/sft.yaml