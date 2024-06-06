# encoding = "utf-8"
from backbone_llm import llm_generator
import json
from tqdm import tqdm
import time
import re
import os
import argparse
import ast
from utils import *
import csv
from prompts import EVAL_SYSTEM_RPOMPT, EVAL_DIFFICULT_TURN_PROMPT

def make_up_dialogue_turns(dialogue):

    dialogue_str = ""
    for i in range(0, len(dialogue), 2):
        dialogue_str += "\n- Turn: " + str(i//2) + "\n"
        dialogue_str += json.dumps(dialogue[i:i+2], indent=2)
        dialogue_str += "\n"

    return dialogue_str.strip()

'''Evaluate the generated dialogues'''
def turn_recognition(api_key, output_dir, target_model, file_suffix, output_file_name):

    judger_args = {"api_key": api_key,
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 200}
    file_suffix = eval(file_suffix)
    output_file = open(os.path.join(output_dir, output_file_name), "a+", encoding="utf-8")

    turn_num_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    turn_num_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

    sample_dirs = os.listdir(output_dir)
    sample_dirs = sorted(sample_dirs)
    print(len(sample_dirs))
    for i, sample_dir in enumerate(sample_dirs):
        
        for suffix in file_suffix:

            # load dialogues
            target_file_name = target_model + suffix +".json"
            target_file = os.path.join(output_dir, sample_dir, target_file_name)
            target_file = json.load(open(target_file, "r"))

            dialogue = target_file["dialogue"]
            dialogue_str = make_up_dialogue_turns(dialogue)

            messages = []
            messages.append({"role": "system", "content": EVAL_SYSTEM_RPOMPT})
            messages.append({"role": "user", "content": EVAL_DIFFICULT_TURN_PROMPT.format(
                DIALOGUE=dialogue_str,
                DIALOGUE_LEN=len(dialogue)
            )})

            turn_num = None
            while turn_num == None:
                try:
                    judgement = llm_generator(messages[:], **judger_args)
                    print(judgement)
                    match = re.search(turn_num_pattern, judgement)
                    if not match:
                        match = re.search(turn_num_pattern_backup, judgement)
                    if match:
                        tmp = ast.literal_eval(match.groups()[0])
                        # print(judgement)
                        turn_num = tmp
                    
                except Exception as e:
                    print(e)
                    continue

            Color.print((turn_num, type(turn_num)), color="YELLOW")
            output_file.write(
                json.dumps({
                    "id":sample_dir,
                    "file_name": target_file_name,
                    "turn_num": turn_num,
                    "judgement": judgement
                })+"\n"
            )

    output_file.close()
    return

'''Collect challenging scripts'''
def script_collection(output_dir, turns_file_name):

    all_script_based_file = open(os.path.join(output_dir, "script_based_all.jsonl"), "a+", encoding="utf-8")
    hard_script_based_file = open(os.path.join(output_dir, "script_based_hard.jsonl"), "a+", encoding="utf-8")
    first_script_based_file = open(os.path.join(output_dir, "script_based_firstchan.jsonl"), "a+", encoding="utf-8")
    later_script_based_file = open(os.path.join(output_dir, "script_based_subseqchan.jsonl"), "a+", encoding="utf-8")
    end_script_based_file = open(os.path.join(output_dir, "script_based_lastonly.jsonl"), "a+", encoding="utf-8")
    cur_count = 0

    with open(os.path.join(output_dir, turns_file_name), "r") as f:
        for line in f:
            line = json.loads(line.strip())

            id = line["id"]
            file_name = line["file_name"]
            turn_num = line["turn_num"]

            target_file_name = os.path.join(output_dir, id, file_name)
            target_file = json.load(open(target_file_name, "r"))
            dialogue = target_file["dialogue"]

            if turn_num == 0:
                
                continue
            
            else:
                
                script_info = {
                    "id": str(cur_count),
                    "source_id": id,
                    "act": target_file["act"],
                    "task_description": target_file["task_description"],
                    "type": target_file["type"],
                    "source_file_name": file_name,
                    "turn_num": turn_num,
                    "dialogue_history": dialogue[:turn_num*2-1],
                }
                cur_count += 1
                all_script_based_file.write(json.dumps(script_info)+"\n")
                first_script_based_file.write(json.dumps(script_info)+"\n")
                hard_script_based_file.write(json.dumps(script_info)+"\n")

                for tn in range(turn_num+1, len(dialogue)//2+1, 1):
                    script_info = {
                        "id": str(cur_count),
                        "source_id": id,
                        "act": target_file["act"],
                        "task_description": target_file["task_description"],
                        "type": target_file["type"],
                        "source_file_name": file_name,
                        "turn_num": tn,
                        "dialogue_history": dialogue[:tn*2-1]
                    }
                    cur_count += 1
                    all_script_based_file.write(json.dumps(script_info)+"\n")
                    later_script_based_file.write(json.dumps(script_info)+"\n")
                    hard_script_based_file.write(json.dumps(script_info)+"\n")

    with open(os.path.join(output_dir, turns_file_name), "r") as f:
        for line in f:
            line = json.loads(line.strip())

            id = line["id"]
            file_name = line["file_name"]
            turn_num = line["turn_num"]

            target_file_name = os.path.join(output_dir, id, file_name)
            target_file = json.load(open(target_file_name, "r"))
            dialogue = target_file["dialogue"]

            if turn_num == 0:
                script_info = {
                    "id": str(cur_count),
                    "source_id": id,
                    "act": target_file["act"],
                    "task_description": target_file["task_description"],
                    "type": target_file["type"],
                    "source_file_name": file_name,
                    "turn_num": turn_num,
                    "dialogue_history": dialogue[:-1],
                }
                cur_count += 1
                end_script_based_file.write(json.dumps(script_info)+"\n")
                all_script_based_file.write(json.dumps(script_info)+"\n")

            else:
                continue
    
    all_script_based_file.close()
    hard_script_based_file.close()
    first_script_based_file.close()
    later_script_based_file.close()
    end_script_based_file.close()



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="challenging script collection")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API_KEY")
    parser.add_argument("--output_dir", type=str, default="./output", help="")
    parser.add_argument("--target_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--file_suffix", type=str, default=str(["", "-1", "-2"]))
    parser.add_argument("--output_file_name", type=str, default="challenging_turns.jsonl")
    args = parser.parse_args()

    turn_recognition(api_key=args.api_key, 
                     output_dir=args.output_dir, 
                     target_model=args.target_model, 
                     file_suffix=args.file_suffix, 
                     output_file_name=args.output_file_name)
    
    script_collection(output_dir=args.output_dir, 
                      turns_file_name=args.output_file_name)

