python3 ./src/lm_judge.py --api_key $API_KEY \
    --target_model Mixtral-8x7B-Instruct-v0.1 \
    --ref_model gpt-4-0125-preview \
    --output_dir "./output" \
    --mode "pairwise" \
    --direction "both" \
    --turn_num 4 \
    --filtered_path "./data/filtered_samples_43.json" \
    --filter_flag "keep"
