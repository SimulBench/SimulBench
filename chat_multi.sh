python3 ./src/interactive_eval.py --character_api_key $API_KEY \
    --character_model /path/to/Llama-2-70B-chat-hf \
    --character_template_name "llama-2" \
    --character_max_tokens 1024 \
    --user_api_key $API_KEY \
    --user_model gpt-3.5-turbo \
    --user_max_tokens 300 \
    --test_file_path "SimulBench/SimulBench" \
    --subset "hard" \
    --test_config_file_path "./data/task_specific_config.json" \
    --output_dir "./output" \
    --turn_num 4 \
    --filtered_path "./data/filtered_samples_43.json" \
    --filter_flag keep

#      --test_file_path "./data/prompts.csv" \