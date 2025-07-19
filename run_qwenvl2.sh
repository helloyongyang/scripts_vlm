pip install transformers==4.53.2

python run_qwenvl.py \
--model-path /data/nvme1/yongyang/projects/llmc_plus/Qwen2.5-VL-3B-Instruct \
--question-file turn2_question.jsonl \
--max_new_tokens 1024
