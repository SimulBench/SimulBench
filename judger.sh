python3 ./src/lm_judge.py --api_key $API_KEY \
    --target_model Llama-2-70b-chat-hf \
    --output_dir "./output" \
    --mode "scoring" \
    --turn_num 4 \
    --filtered_path "./data/hard_subset.json" \
    --filter_flag "keep"



