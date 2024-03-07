# encoding = "utf-8"
from backbone_llm import llm_generator
import json
import argparse
import ast
import re
import time
import os
from utils import *
from prompts import EVAL_PROMPT_SCORING_REF, EVAL_RPOMPT_PAIRWISE, EVAL_SYSTEM_RPOMPT, EVAL_PROMPT_SCORING
import random


def model_scoring(target_model, ref_model, output_dir, judger_args, mode, filtered_path=None, filter_flag=None,
                  utt_num=100):
    # test_samples = load_test_cases(test_file_path)
    if filtered_path != None:
        filtered_dirs = json.load(open(filtered_path, "r"))

    if filter_flag == "keep":
        sample_dirs = filtered_dirs
    else:
        sample_dirs = os.listdir(output_dir)
        sample_dirs = sorted(sample_dirs)

    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    ratings = []

    for i, sample_dir in enumerate(sample_dirs):

        if filter_flag == "remove":
            if sample_dir in filtered_dirs:
                continue
        print(sample_dir)

        # load dialogues
        target_file = os.path.join(output_dir, sample_dir, target_model + ".json")
        target_file = json.load(open(target_file, "r"))
        simulation = target_file["act"]
        target_dialogue_list = target_file["dialogue"][:utt_num]
        target_dialogue = convert_dialogue_list_to_string(target_dialogue_list,
                                                          role_dict={"user": "User", "assistant": "Assistant"})

        ref_dialogue_list = None
        ref_dialogue = None
        if mode == "scoring_ref":
            ref_file = os.path.join(output_dir, sample_dir, ref_model + ".json")
            ref_dialogue_list = json.load(open(ref_file, "r"))["dialogue"]
            ref_dialogue = convert_dialogue_list_to_string(ref_dialogue_list,
                                                           role_dict={"user": "User", "assistant": "Assistant"})

        output_file_name = os.path.join(output_dir, sample_dir, mode + "_" + target_model + "_" + str(
            len(target_dialogue_list) // 2) + "-gpt4.json")
        if os.path.exists(output_file_name):
            print("already finished")
            continue
        output_file = open(output_file_name, "w", encoding="utf-8")

        # construct the input messages
        if mode == "scoring":

            messages = []
            messages.append({"role": "system", "content": EVAL_SYSTEM_RPOMPT})
            messages.append({"role": "user", "content": EVAL_PROMPT_SCORING.format(
                SIMULATION=simulation,
                DIALOGUE=target_dialogue
            )})
        else:

            messages = []
            messages.append({"role": "system", "content": EVAL_SYSTEM_RPOMPT})
            messages.append({"role": "user", "content": EVAL_PROMPT_SCORING_REF.format(
                SIMULATION=simulation,
                REFERENCE=ref_dialogue,
                DIALOGUE=target_dialogue
            )})

        # collect the rating
        rating = None
        while rating == None:
            try:
                judgement = llm_generator(messages[:], **judger_args)
                match = re.search(one_score_pattern, judgement)
                if not match:
                    match = re.search(one_score_pattern_backup, judgement)
                if match:
                    tmp = ast.literal_eval(match.groups()[0])
                    print(judgement)
                    rating = tmp
            except Exception as e:
                print(e)
                continue

        print(rating, type(rating))
        ratings.append(float(rating))

        json.dump({
            "id": sample_dir,
            "score": rating,
            "judgement": judgement,
            "target_model": target_model,
            "dialogue": target_dialogue_list,
            "ref_model": ref_model,
            "dialogue_ref": ref_dialogue_list,
            "messages": messages
        }, output_file, indent=4)
        output_file.close()
    print(ratings)
    print(sum(ratings) / len(ratings))
    return


def pair_data_comparison(dialogue_1, dialogue_2, simulation, judger_args, direction_mode="12"):
    winner_12, winner_21 = "error", "error"
    judgement_12, messages_12, judgement_21, messages_21 = None, None, None, None

    messages = []
    messages.append({"role": "system", "content": EVAL_SYSTEM_RPOMPT})  #

    if direction_mode == "12" or direction_mode == "both":
        # model1, model2
        messages.append({"role": "user", "content": EVAL_RPOMPT_PAIRWISE.format(
            SIMULATION=simulation,
            DIALOGUE_1=dialogue_1,
            DIALOGUE_2=dialogue_2)})
        messages_12 = messages[:]
        print(num_tokens_from_messages(messages_12, "gpt-4"))

        while winner_12 == "error":
            print("12")
            try:
                judgement_12 = llm_generator(messages_12, **judger_args)
                if "[[A]]" in judgement_12:
                    winner_12 = "A"
                elif "[[B]]" in judgement_12:
                    winner_12 = "B"
                elif "[[C]]" in judgement_12:
                    winner_12 = "tie"
                else:
                    winner_12 = "error"
            except Exception as e:
                print(e)
                continue

        messages.pop()

    if direction_mode == "21" or direction_mode == "both":

        # model2, model1
        messages.append({"role": "user", "content": EVAL_RPOMPT_PAIRWISE.format(
            SIMULATION=simulation,
            DIALOGUE_1=dialogue_2,
            DIALOGUE_2=dialogue_1)})
        messages_21 = messages[:]
        print(num_tokens_from_messages(messages_21, "gpt-4"))

        while winner_21 == "error":
            print("21")
            try:
                judgement_21 = llm_generator(messages_21, **judger_args)
                if "[[A]]" in judgement_21:
                    winner_21 = "A"
                elif "[[B]]" in judgement_21:
                    winner_21 = "B"
                elif "[[C]]" in judgement_21:
                    winner_21 = "tie"
                else:
                    winner_21 = "error"
            except Exception as e:
                print(e)
                continue

    return winner_12, judgement_12, messages_12, winner_21, judgement_21, messages_21


def model_comparison(target_model, ref_model, output_dir, judger_args, mode, filtered_path=None, filter_flag=None,
                     utt_num=100, direction="random"):
    if filtered_path != None:
        filtered_dirs = json.load(open(filtered_path, "r"))

    if filter_flag == "keep":
        sample_dirs = filtered_dirs
    else:
        sample_dirs = os.listdir(output_dir)
        sample_dirs = sorted(sample_dirs)

    score = {"A": 0, "B": 0, "tie": 0}
    print(len(sample_dirs))

    for i, sample_dir in enumerate(sample_dirs):
        if filter_flag == "remove":
            if sample_dir in filtered_dirs:
                continue
        direction_mode = None

        if direction == "random":
            if random.random() < 0.5:
                direction_mode = "12"
            else:
                direction_mode = "21"
        elif direction == "both":
            direction_mode = "both"

        print(sample_dir, direction_mode)

        target_file = os.path.join(output_dir, sample_dir, target_model + ".json")
        target_file = json.load(open(target_file, "r"))
        target_dialogue_list = target_file["dialogue"][:utt_num]
        simulation = target_file["act"]
        target_dialogue = convert_dialogue_list_to_string(target_dialogue_list,
                                                          role_dict={"user": "User", "assistant": "Assistant"})

        ref_file = os.path.join(output_dir, sample_dir, ref_model + ".json")
        ref_dialogue_list = json.load(open(ref_file, "r"))["dialogue"][:utt_num]
        ref_dialogue = convert_dialogue_list_to_string(ref_dialogue_list,
                                                       role_dict={"user": "User", "assistant": "Assistant"})

        output_file_name = os.path.join(output_dir, sample_dir,
                                        mode + "_" + ref_model + "_vs_" + target_model + "_" + str(
                                            len(target_dialogue_list) // 2) + ".json")
        if os.path.exists(output_file_name):
            print("already finished")
            continue
        output_file = open(output_file_name, "w", encoding="utf-8")

        winner_12, judgement_12, messages_12, winner_21, judgement_21, messages_21 = pair_data_comparison(
            dialogue_1=ref_dialogue,
            dialogue_2=target_dialogue,
            simulation=simulation,
            judger_args=judger_args,
            direction_mode=direction_mode
        )

        final_winner = None

        if direction_mode == "12":
            final_winner = winner_12
            json.dump({
                "id": sample_dir,
                "final_winner": final_winner,
                "target_model": target_model,
                "dialogue": target_dialogue_list,
                "ref_model": ref_model,
                "dialogue_ref": ref_dialogue_list,
                "winner": winner_12,
                "judgement": judgement_12,
                "messages": messages_12,
                "direction_mode": direction_mode
            }, output_file, indent=4)

        elif direction_mode == "21":
            if winner_21 == "A":
                final_winner = "B"
            elif winner_21 == "B":
                final_winner = "A"
            else:
                final_winner = "tie"

            json.dump({
                "id": sample_dir,
                "final_winner": final_winner,
                "target_model": target_model,
                "dialogue": target_dialogue_list,
                "ref_model": ref_model,
                "dialogue_ref": ref_dialogue_list,
                "winner": winner_21,
                "judgement": judgement_21,
                "messages": messages_21,
                "direction_mode": direction_mode
            }, output_file, indent=4)


        elif direction_mode == "both":
            if winner_12 == "A" and winner_21 == "B":
                final_winner = "A"
            elif winner_12 == "B" and winner_21 == "A":
                final_winner = "B"
            else:
                final_winner = "tie"

            json.dump({
                "id": sample_dir,
                "final_winner": final_winner,
                "winner_12": winner_12,
                "judgement_12": judgement_12,
                "winner_21": winner_21,
                "judgement_21": judgement_21,
                "target_model": target_model,
                "dialogue": target_dialogue_list,
                "ref_model": ref_model,
                "dialogue_ref": ref_dialogue_list,
                "messages_12": messages_12,
                "messages_21": messages_21
            }, output_file, indent=4)

        score[final_winner] += 1

        output_file.close()
        time.sleep(3)
    print(score)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="persona chat evaluation")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API_KEY")
    # parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1") 
    # parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    # parser.add_argument("--temperature", type=float, default=0)
    # parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--filtered_path", type=str, default=None, help="a specific group of sample ids")
    parser.add_argument("--filter_flag", type=str, default=None, help="keep or remove the ids in filtered_path")
    parser.add_argument("--ref_model", type=str, default=None, help="the name of a model, which is the first model in pairwise comparison or the reference model in scoring with a reference.")
    parser.add_argument("--target_model", type=str, default="Llama-2-70b-chat-hf", help="the name of a model to be evaluated.")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--mode", type=str, help="scoring, scoring_ref, or pairwise")
    parser.add_argument("--direction", type=str, default="random",
                        help="a random direction or both directions for pairwise comparison")
    parser.add_argument("--turn_num", type=int, default=10,
                        help="the number of turns to be evaluated. utt_num = turn_num *2")
    args = parser.parse_args()

    random.seed(42)
    judger_args = {"api_key": args.api_key,
                   "base_url": "https://api.openai.com/v1",
                   "model": "gpt-4",
                   "temperature": 0.0,
                   "max_tokens": 1024}

    if args.mode == "scoring" or args.mode == "scoring_ref":
        model_scoring(args.target_model, args.ref_model, args.output_dir, judger_args, args.mode,
                      filtered_path=args.filtered_path, filter_flag=args.filter_flag, utt_num=args.turn_num * 2)
    elif args.mode == "pairwise":
        model_comparison(args.target_model, args.ref_model, args.output_dir, judger_args, args.mode,
                         filtered_path=args.filtered_path, filter_flag=args.filter_flag, utt_num=args.turn_num * 2,
                         direction=args.direction)
