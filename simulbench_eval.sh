python3 ./src/script_based_eval.py --judger_api_key $API_KEY \
    --character_api_key $API_KEY \
    --character_model gpt-4o-2024-05-13 \
    --character_template_name "" \
    --character_max_tokens 1024 \
    --character_base_url https://api.openai.com/v1 \
    --test_file_path "./output/script_based_all.jsonl" \
    --output_dir "./output_script" 
