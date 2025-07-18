pip install transformers -U

python run_qwenvl.py \
--model-path /data/nvme1/yongyang/projects/llmc_plus/Qwen2.5-VL-3B-Instruct \
--question-file turn1_question.jsonl \
--max_new_tokens 1024
