pip install transformers==4.53.2

python run_qwenvl.py \
--model-path /data/nvme1/yongyang/projects/llmc_plus/Qwen2.5-VL-7B-Instruct \
--question-file ../data/turn1_question.jsonl \
--images-dir ../data/images \
--max_new_tokens 1024
