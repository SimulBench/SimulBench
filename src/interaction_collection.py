# encoding = "utf-8"
from backbone_llm import llm_generator, ConversationPipeline
import json
from tqdm import tqdm
import time
import re
import os
import argparse
from utils import *
import csv
from prompts import USER_PROMPT


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
        task_file_path,
        output_dir,
        user_api_key=None,
        user_base_url=None,
        user_model=None,
        subset=None,
        user_temperature=0.0,
        user_max_tokens=500,
        character_temperature=0.0,
        character_max_tokens=500,
        turn_num=2
):
    print(user_max_tokens, character_max_tokens)

    test_samples = load_test_cases(task_file_path, subset=subset)
    Color.print("loaded {} samples".format(len(test_samples)), color="GREEN")

    system_prompt = "You are a helpful assistant."

    for i, sample in tqdm(enumerate(test_samples), desc="Character"):
        
        id = sample["id"]
        
        Color.print(f"id: {id}", "YELLOW")

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_path = os.path.join(output_dir, id)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_file_path = os.path.join(output_path, character_model.split("/")[-1]) + "-2.json"
        user_generations = [] # record the generated user strategies and request
        character_messages = [] # record the character_messages
        dialogue = [] # record the current dialogue
        if os.path.exists(output_file_path):
            try:
                tmp = json.load(open(output_file_path, "r"))
                user_generations = tmp["user_generations"]
                character_messages = tmp["character_messages"]
                dialogue = tmp["dialogue"]
            except Exception as e:
                print(e)
                pass
        output_file = open(output_file_path, "w", encoding="utf-8")

        first_user_utterance = sample["task_description"]
        request_type = sample["request_type"]
        strategy_type = "a short description of the adopted strategy"
       
        dialogue_success = False
        while dialogue_success == False:
            try:
                
                max_turn_num = turn_num
                cur_turn_num = len(dialogue) // 2

                # set the original/designed system prompts
                if len(character_messages) == 0:
                    character_messages.append({"role": "system", "content": system_prompt})

                # begin the dialogue
                while cur_turn_num < max_turn_num: #and stop_flag == False:
                    
                    print(cur_turn_num)

                    # user's turn
                    if len(dialogue)==0 or dialogue[-1]["role"]!="user":

                        if cur_turn_num == 0 and first_user_utterance != None:
                            user_utterance = first_user_utterance
                            Color.print(f"user: {user_utterance}", "RED")

                        else:
                            cur_content = USER_PROMPT.replace("{DIALOGUE}", make_up_dialogue(list(dialogue)), 1).replace(
                                "{REQUEST_TYPE}", request_type, 1).replace("{STRATEGY_TYPE}", strategy_type, 1)
                            
                            cur_messages = [{"role": "system", "content": system_prompt},{"role":"user","content":cur_content}]
                            Color.print(cur_content, "YELLOW")
                            
                            user_utterance = llm_generator(messages=cur_messages, api_key=user_api_key,
                                                        base_url=user_base_url, temperature=user_temperature,
                                                        model=user_model, max_tokens=user_max_tokens)
                            time.sleep(1)
                            Color.print(f"user: {user_utterance}", "RED")

                            try:
                                user_generation = json.loads(user_utterance)
                                user_utterance = user_generation["user"]["request"]
                            except:
                                if user_utterance.startswith("```\n"):
                                    user_utterance = user_utterance[4:]
                                if user_utterance.endswith("\n```"):
                                    user_utterance = user_utterance[:-4]
                                user_generation = json.loads(user_utterance)
                                user_utterance = user_generation["user"]["request"]
                            
                            assert len(user_utterance) != 0
                            user_generations.append(user_generation)


                        dialogue.append({"role": "user", "content": user_utterance})
                        character_messages.append({"role": "user", "content": user_utterance})

                
                    # assistant's turn
                    if dialogue[-1]["role"]!="assistant":

                        character_utterance = llm_generator(messages=character_messages[:], api_key=character_api_key,
                                                            base_url=character_base_url, temperature=character_temperature,
                                                            model=character_model, max_tokens=character_max_tokens)

                        character_utterance = utterance_normalization(character_utterance, model_name=character_model)
                        time.sleep(1)

                        Color.print(f"character: {character_utterance}", "BLUE")
                        dialogue.append({"role": "assistant", "content": character_utterance})
                        character_messages.append({"role": "assistant", "content": character_utterance})

                    cur_turn_num += 1
                assert len(dialogue) == max_turn_num * 2
                dialogue_success = True
            except Exception as e:
                print(e)
                continue

        dialogues = {
            "id": sample["id"],
            "act": sample["act"],
            "task_description": sample["task_description"],
            "type": sample["type"],
            "dialogue": dialogue,
            "character_messages": character_messages,
            "user_generations": user_generations,

        }
        json.dump(dialogues, output_file, indent=4)
        output_file.close()
        time.sleep(1)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="simulation conversation collection")
    parser.add_argument("--character_api_key", type=str, default="", help="OpenAI API_KEY")
    parser.add_argument("--character_model", type=str, default="Llama-2-70b-chat-hf", help="The path to an LLM playing the role of a simulator")
    parser.add_argument("--character_max_tokens", type=int, default=200, help="The max number of tokens to be generated by the simulator")
    parser.add_argument("--character_temperature", type=float, default=1.0)
    parser.add_argument("--user_api_key", type=str, default="", help="OpenAI API_KEY")
    parser.add_argument("--user_model", type=str, default="gpt-3.5-turbo", help="The path to an LLM playing the role of the user agent")
    parser.add_argument("--user_max_tokens", type=int, default=200, help="The max number of tokens to be generated by the user agent")
    parser.add_argument("--user_temperature", type=float, default=1.0)

    parser.add_argument("--task_file_path", type=str, default="SimulBench/SimulBench-tasks", help="simulation tasks, load from huggingface/datasets")
    parser.add_argument("--subset", type=str, default=None, help="")
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
    parser.add_argument("--turn_num", type=int, default=4, help="the number of turns to be collected")
    args = parser.parse_args()

   
    args.character_base_url = "https://api.openai.com/v1"
    args.user_base_url = "https://api.openai.com/v1"

    conversation_generation(
        character_api_key=args.character_api_key,
        character_base_url=args.character_base_url,
        character_model=args.character_model,
        character_max_tokens=args.character_max_tokens,
        character_temperature=args.character_temperature,

        user_api_key=args.user_api_key,
        user_max_tokens=args.user_max_tokens,
        user_model=args.user_model,
        user_base_url=args.user_base_url,
        user_temperature=args.user_temperature,

        task_file_path=args.task_file_path,
        subset=args.subset,
        output_dir=args.output_dir,
        turn_num=args.turn_num
    )
