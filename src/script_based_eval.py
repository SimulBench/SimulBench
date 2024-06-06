# encoding = "utf-8"
from backbone_llm import llm_generator, ConversationPipeline
import json
from tqdm import tqdm
import time
import re
import os
import argparse
import ast
from utils import *
import csv
from prompts import EVAL_SYSTEM_RPOMPT, EVAL_PROMPT_SCORING
from fastchat.conversation import get_conv_template

def utterance_normalization(text, model_name):

    text = text.strip()
    if text.startswith("\""):
        text = text[1:]
        if text.endswith("\""):
            text = text[:-1]

    return text

def conversation_generation(
        character_api_key,
        character_base_url,
        character_model,
        test_file_path,
        subset,
        output_dir,
        character_temperature=0.0,
        character_max_tokens=500,
        character_system_prompt="You are a helpful assistant.",
        judger_args=None
):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    test_samples = load_test_cases(test_file_path, subset=subset)

    # test_samples = []
    # with open(test_file_path, "r") as f:
    #     for line in f:
    #         line = json.loads(line.strip())
    #         test_samples.append(line)
    
    
    for i, sample in tqdm(enumerate(test_samples), desc="Character"):
        
        sample_dir = os.path.join(output_dir, sample["id"]+"_"+sample["source_id"])
        
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        output_file_path = os.path.join(sample_dir, character_model.split("/")[-1]+".json")
        if os.path.exists(output_file_path):
            continue
        output_file = open(output_file_path, "w", encoding="utf-8")

        dialogue_history = sample["dialogue_history"]
        character_messages = [{"role": "system", "content": character_system_prompt}]
        character_messages += dialogue_history

        # assistant's turn generation
        if type(character_base_url) != str:
            character_messages.append({"role": "assistant", "content": None})

        response_success = False
        character_utterance = None
        while response_success == False:
                
            character_utterance = llm_generator(messages=character_messages[:], api_key=character_api_key,
                                                base_url=character_base_url, temperature=character_temperature,
                                                model=character_model, max_tokens=character_max_tokens)
            
            character_utterance = utterance_normalization(character_utterance, model_name=character_model)
            Color.print(f"character: {character_utterance}", "BLUE")

            response_success = True
            time.sleep(1)

        if type(character_base_url) != str:
            character_messages.pop()

        sample_output = {
            "id":sample["id"],
            "source_id":sample["source_id"],
            "act":sample["act"],
            "task_description":sample["task_description"],
            "type":sample["type"],
            "source_file":sample["source_file_name"],
            "turn_num":sample["turn_num"],
            "dialogue_history":sample["dialogue_history"],
            "response":{"role":"assistant", "content":character_utterance},
            "character_system_prompt":character_system_prompt,
        }

        # assistant's turn evaluation
        if judger_args!=None:
            one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
            one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
            
            judger_messages = [{"role":"system", "content": EVAL_SYSTEM_RPOMPT},
                               {"role":"user", "content": EVAL_PROMPT_SCORING.format(SIMULATION=sample["act"], DIALOGUE=make_up_dialogue(dialogue_history), RESPONSE=json.dumps({"AI": character_utterance}, indent=2))}]

            # collect the rating    
            rating = None
            while rating == None:
                try:
                    judgement = llm_generator(judger_messages, **judger_args)
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
            sample_output["score"] = rating
            sample_output["judgement"] = judgement
        
        json.dump(sample_output, output_file, indent=4)
        output_file.close()
        time.sleep(1)
        # exit(0)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="simulation conversation collection")
    parser.add_argument("--judger_api_key", type=str, default=None)
    parser.add_argument("--character_api_key", type=str, default=None, help="OpenAI API_KEY")
    parser.add_argument("--character_model", type=str, default="gpt-4-0125-preview", help="The path to an LLM playing the role of a simulator")
    parser.add_argument("--character_template_name", type=str, default=None, help="The chat template to be used for open LLMs. See FastChat.conversation for more details.")
    parser.add_argument("--character_max_tokens", type=int, default=1024, help="The max number of tokens to be generated by the simulator")
    parser.add_argument("--character_base_url", type=str, default="https://api.openai.com/v1", help="https://api.openai.com/v1, https://api.together.xyz/v1, local")
    
    parser.add_argument("--test_file_path", type=str, default="./output/script_based_all.jsonl", help="simulation tasks, load from the local file or from huggingface/datasets by SimulBench/SimulBench")
    parser.add_argument("--subset", type=str, default="all", help="all, hard, firstchan, subseqchan, lastonly, stateful, stateless")

    parser.add_argument("--output_dir", type=str, default="./output_script", help="output directory")
    args = parser.parse_args()

    # if args.character_model in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview"]: # openai-api
    #     args.character_base_url = "https://api.openai.com/v1"
    # elif args.character_model in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"]: # togetherai-api
    #     args.character_base_url = "https://api.together.xyz/v1"

    if args.character_base_url == "local": # local model
        args.character_base_url = ConversationPipeline(args.character_model, args.character_template_name)


    if args.character_template_name:
        character_system_prompt = get_conv_template(args.character_template_name).system_message # original prompt
    else:
        character_system_prompt = "You are a helpful assistant."

    if args.judger_api_key:
        judger_args = {"api_key": args.judger_api_key,
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 1024}
    else:
        judger_args = None
    
    conversation_generation(
        character_api_key=args.character_api_key,
        character_base_url=args.character_base_url,
        character_model=args.character_model,
        character_max_tokens=args.character_max_tokens,

        test_file_path=args.test_file_path,
        subset=args.subset,
        output_dir=args.output_dir,
        character_system_prompt=character_system_prompt,
        judger_args=judger_args
    )

