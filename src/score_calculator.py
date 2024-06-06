# encoding = "utf-8"
import json
import os
import argparse
# from collections import defaultdict
from datasets import load_dataset
from utils import load_test_cases


def main(args):

    scores = []

    sample_dirs = []
    test_samples = load_test_cases(args.test_file_path, args.subset)
    for test_sample in test_samples:
        sample_dirs.append(test_sample["id"]+"_"+test_sample["source_id"])

    if args.output_dir:
        for sample_dir in sample_dirs:  

            with open(os.path.join(args.output_dir, sample_dir, args.model_name+".json"), "r") as f:
                content = json.load(f)
                scores.append(content["score"])

    else:

        outputs = load_dataset("SimulBench/SimulBench-results", args.model_name, split="test")
        for output in outputs:
            if str(output["id"]) + "_" + output["source_id"] in sample_dirs:
                scores.append(output["score"])
    
    print(f"Avg score of {args.model_name} on {len(scores)} samples is {sum(scores)/len(scores)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="simulation score")
    parser.add_argument("--model_name", type=str, default="Llama-2-13b-chat-hf", help="model name")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory, None for loading results from huggingface datasets")
    parser.add_argument("--test_file_path", type=str, default="SimulBench/SimulBench", help="simulation tasks, load from huggingface/datasets by SimulBench/SimulBench")
    parser.add_argument("--subset", type=str, default="all", help="all, hard, firstchan, subseqchan, lastonly, stateful, stateless")
    args = parser.parse_args()

    main(args)
