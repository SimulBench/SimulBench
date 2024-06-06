python3 ./src/lm_judge.py --api_key $API_KEY \
    --target_model Mixtral-8x7B-Instruct-v0.1 \
    --ref_model gpt-4-0125-preview \
    --output_dir "./output_script" \
    --test_file_path "SimulBench/SimulBench" \
    --mode "pairwise" \
    --direction "both"
