pip install transformers==4.45.2

python run_llava.py \
--model-path /data/nvme1/yongyang/projects/llmc_plus/llava-v1.5-7b \
--question-file ../data/turn2_question.jsonl \
--images-dir ../data/images \
--max_new_tokens 1024
