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
    if "chatglm" in model_name.lower():
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>", 1)[0]
        if "<|user|>" in text:
            text = text.split("<|user|>", 1)[0]

    text = text.strip()
    if text.startswith("\""):
        text = text[1:]
        if text.endswith("\""):
            text = text[:-1]

    return text


def make_up_dialogue(utterances):
    print(utterances)
    outputs = []
    for utt in utterances:
        if utt["role"] == "user":
            outputs.append({"user": utt["content"]})
        else:
            outputs.append({"AI": utt["content"]})
    return json.dumps(outputs, indent=2)


def conversation_generation(
        character_api_key,
        character_base_url,
        character_model,
        test_file_path,
        test_config_file_path,
        output_dir,
        user_api_key=None,
        user_base_url=None,
        user_model=None,
        filtered_path=None,
        filter_flag=None,
        subset=None,
        user_temperature=0.0,
        user_max_tokens=500,
        character_temperature=0.0,
        character_max_tokens=500,
        turn_num=2
):
    print(user_max_tokens, character_max_tokens)

    test_samples = load_test_cases(test_file_path, subset=subset)
    test_config = json.load(open(test_config_file_path, "r"))

    if filtered_path != None:
        filtered_samples = json.load(open(filtered_path, "r"))
        print("filtered_samples count:", len(filtered_samples))

    if type(character_base_url) != str:
        character_prompt = character_base_url.conv_template.system_message  # original prompt
    else:
        character_prompt = "You are a helpful assistant."

    for i, sample in tqdm(enumerate(test_samples), desc="Character"):

        if subset==None:
            id = str(i).zfill(3) + "#" + sample["act"].replace("/", "-").replace(" ", "_")
        else:
            id = sample["id"]

        if filtered_path != None:
            if filter_flag == "keep" and id not in filtered_samples:
                continue
            elif filter_flag == "remove" and id in filtered_samples:
                continue

        sample_config = test_config[id] if id in test_config else {}

        Color.print(f"id: {id}", "YELLOW")

        output_path = os.path.join(output_dir, id)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_file_path = os.path.join(output_path, character_model.split("/")[-1]) + ".json"
        init_user_generations = []
        init_character_messages = []
        init_dialogue = []
        if os.path.exists(output_file_path):
            try:
                tmp = json.load(open(output_file_path, "r"))
                init_user_generations = tmp["user_generations"]
                init_character_messages = tmp["character_messages"]
                init_dialogue = tmp["dialogue"]
            except Exception as e:
                print(e)
                pass
        output_file = open(output_file_path, "w", encoding="utf-8")

        first_user_utterance = sample["task_description"]
        print(first_user_utterance)
        request_type = sample_config[
            "request_type"] if "request_type" in sample_config else "your executable command or instruction"
        strategy_type = sample_config[
            "strategy_type"] if "strategy_type" in sample_config else "[a short description of the adopted strategy]"

        dialogue_success = False
        while dialogue_success == False:
            try:
                character_messages = init_character_messages
                user_generations = init_user_generations
                dialogue = init_dialogue  # will save the generated utterances in init_dialogue when the api failed
                max_turn_num = sample_config[
                    "max_turn_num"] if "max_turn_num" in sample_config.keys() else turn_num  # 6
                cur_turn_num = len(init_dialogue) // 2
                stop_flag = False

                # set the original/designed system prompts
                if len(character_messages) == 0:
                    character_messages.append({"role": "system", "content": character_prompt})

                # begin the dialogue
                while cur_turn_num < max_turn_num and stop_flag == False:

                    # user's turn
                    print(cur_turn_num)
                    if cur_turn_num == 0 and first_user_utterance != None:
                        user_utterance = first_user_utterance
                        Color.print(f"user: {user_utterance}", "RED")

                    else:
                        cur_content = USER_PROMPT.replace("{DIALOGUE}", make_up_dialogue(list(dialogue)), 1).replace(
                            "{REQUEST_TYPE}", request_type, 1).replace("{STRATEGY_TYPE}", strategy_type, 1)
                        cur_messages = [{"role": "system", "content": cur_content}]
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

                        user_generations.append(user_generation)

                    if len(user_utterance) != 0 or stop_flag != True:
                        dialogue.append({"role": "user", "content": user_utterance})
                    character_messages.append({"role": "user", "content": user_utterance})

                    if stop_flag:
                        break

                    # assistant's turn
                    if type(character_base_url) != str:
                        character_messages.append({"role": "assistant", "content": None})

                    character_utterance = llm_generator(messages=character_messages[:], api_key=character_api_key,
                                                        base_url=character_base_url, temperature=character_temperature,
                                                        model=character_model, max_tokens=character_max_tokens)
                    if type(character_base_url) != str:
                        character_messages.pop()

                    character_utterance = utterance_normalization(character_utterance, model_name=character_model)
                    time.sleep(1)

                    Color.print(f"character: {character_utterance}", "BLUE")
                    dialogue.append({"role": "assistant", "content": character_utterance})
                    character_messages.append({"role": "assistant", "content": character_utterance})

                    cur_turn_num += 1
                dialogue_success = True
            except Exception as e:
                print(e)
                continue

        dialogues = {
            "id": f"{id}",
            "act": sample["act"],
            "task_description": first_user_utterance,
            "dialogue": dialogue,
            # "user_messages": user_messages[:],
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
    parser.add_argument("--character_template_name", type=str, default="llama-2", help="The chat template to be used for open LLMs. See FastChat.conversation for more details.")
    parser.add_argument("--character_max_tokens", type=int, default=200, help="The max number of tokens to be generated by the simulator")
    # parser.add_argument("--character_base_url", type=str, default=None)

    parser.add_argument("--user_api_key", type=str, default="", help="OpenAI API_KEY")
    parser.add_argument("--user_model", type=str, default="gpt-3.5-turbo", help="The path to an LLM playing the role of the user agent")
    parser.add_argument("--user_template_name", type=str, default="llama-2", help="The chat template to be used for open LLMs. See FastChat.conversation for more details.")
    parser.add_argument("--user_max_tokens", type=int, default=200, help="The max number of tokens to be generated by the user agent")
    # parser.add_argument("--user_base_url", type=str, default=None)

    parser.add_argument("--filtered_path", type=str, default=None, help="a specific group of sample ids")
    parser.add_argument("--filter_flag", type=str, default=None, help="keep or remove the ids in filtered_path")
    parser.add_argument("--test_file_path", type=str, default="../data/prompts.csv", help="simulation tasks")
    parser.add_argument("--subset", type=str, default=None, help="if test_file_path is SimulBench/SimulBench, subset should be one of [all, hard, objective, subjective, system, tool, role], else None.")
    parser.add_argument("--test_config_file_path", type=str, default="./data/task_specific_config.json", help="path to the task specific configurations for the user agent")
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
    parser.add_argument("--turn_num", type=int, default=4, help="the number of turns to be collected")
    args = parser.parse_args()

    if args.character_model not in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview"]:
        args.character_base_url = ConversationPipeline(args.character_model, args.character_template_name)
    else:
        print("here")
        args.character_base_url = "https://api.openai.com/v1"

    if args.user_model not in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview"]:
        if args.user_model == args.character_model:
            args.user_base_url = args.character_base_url
        else:
            args.user_base_url = ConversationPipeline(args.user_model, args.user_template_name)
    else:
        args.user_base_url = "https://api.openai.com/v1"

    conversation_generation(
        character_api_key=args.character_api_key,
        character_base_url=args.character_base_url,
        character_model=args.character_model,
        character_max_tokens=args.character_max_tokens,

        user_api_key=args.user_api_key,
        user_max_tokens=args.user_max_tokens,
        user_model=args.user_model,
        user_base_url=args.user_base_url,

        filtered_path=args.filtered_path,
        filter_flag=args.filter_flag,
        test_file_path=args.test_file_path,
        subset=args.subset,
        test_config_file_path=args.test_config_file_path,
        output_dir=args.output_dir,
        turn_num=args.turn_num
    )
