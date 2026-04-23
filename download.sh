export HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ckpt/Qwen2.5-VL-7B-Instruct

huggingface-cli download nyu-visionx/VSI-Bench --repo-type dataset --local-dir Evaluation/VSIBench
huggingface-cli download lmms-lab/VideoMMMU --repo-type dataset --local-dir Evaluation/VideoMMMU
huggingface-cli download yale-nlp/MMVU --repo-type dataset --local-dir Evaluation/MMVU
huggingface-cli download OpenGVLab/MVBench --repo-type dataset --local-dir Evaluation/MVBench
huggingface-cli download lmms-lab/TempCompass --repo-type dataset --local-dir Evaluation/TempCompass
huggingface-cli download lmms-lab/Video-MME --repo-type dataset --local-dir Evaluation/VideoMME