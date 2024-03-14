# encoding = "utf-8"
from data.obj_subj import *
from data.system_tool_role import *
import json
import os
from utils import load_test_cases
import argparse
# from collections import defaultdict
from datasets import load_dataset



def load_data_from_hf(subset):
    all_data = load_dataset("SimulBench/SimulBench", subset, split="test")
    # samples = {}
    # for data in all_data:
    #     if data["test_model"] not in samples:
    #         samples[data["test_model"]] = {}
    #     samples[data["test_model"]] = data
    return all_data

def load_data_from_local(subset, output_dir):

    all_data = []
    if subset=="hard":
        hard_dirs = json.load(open("./data/hard_subset.json", "r"))
    sample_dirs = os.listdir(output_dir)
    sample_dirs = sorted(sample_dirs)

    for sample_dir in sample_dirs:
        if subset!="all":
            if subset=="hard":
                if sample_dir not in hard_dirs:
                    continue
            elif subset=="objective":
                if sample_dir not in objective_list:
                    continue
            elif subset=="subjective":
                if sample_dir not in subjective_list:
                    continue
            elif subset=="system":
                if sample_dir not in system_list:
                    continue
            elif subset=="tool":
                if sample_dir not in tool_list:
                    continue
            elif subset=="role":
                if sample_dir not in role_list:
                    continue

        file_paths = os.listdir(os.path.join(output_dir, sample_dir))

        for file_path in file_paths:

            if file_path.startswith("scoring") and file_path.endswith("4-gpt4.json"):
                with open(os.path.join(output_dir, sample_dir, file_path), "r") as f:
                    content = json.load(f)

                    all_data.append({"id": content["id"],
                                     "test_model": content["target_model"],
                                     "dialogue": content["dialogue"],
                                     "score": content["score"],
                                     "judgement": content["judgement"]})
    return all_data

def main(args):

    samples = []
    if args.data_source=="hf":
        samples = load_data_from_hf(args.subset)
    elif args.data_source=="local":
        samples = load_data_from_local(args.subset, args.output_dir)
    else:
        print(f"{args.data_source} is unavailable")
        exit(0)

    scores = {}

    for data in samples:
        if data["test_model"] not in scores:
            scores[data["test_model"]] = []
        scores[data["test_model"]].append(data["score"])

    for key in scores:
        assert len(scores[key])==len(samples)//5
        print(f"Micro-Avg score of {key} on the {args.subset} subset is {sum(scores[key])/len(scores[key])}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="simulation score")
    parser.add_argument("--data_source", type=str, default="local", help="local or hf")
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
    parser.add_argument("--subset", type=str, default="all",
                        help="all, hard, objective, subjective, tool, system, role")
    args = parser.parse_args()

    main(args)
