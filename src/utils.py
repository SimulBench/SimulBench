# encoding = utf-8
import csv
import json
import tiktoken
from datasets import load_dataset
from perspective import PerspectiveAPI # https://pypi.org/project/perspective/
import time

def make_up_dialogue(utterances):
    # print(utterances)
    outputs = []
    for utt in utterances:
        if utt["role"] == "user":
            outputs.append({"user": utt["content"]})
        else:
            outputs.append({"AI": utt["content"]})
    return json.dumps(outputs, indent=2)


def dict_to_str(dictionary):
    output_str = ""
    for key in dictionary.keys():
        output_str += "- " + str(key) + ": " + str(dictionary[key]) + "\n"

    return output_str


# ANSI code for different colors
class Color:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'

    @staticmethod
    def print(content, color: str):
        print(f"{getattr(Color, color.upper())}{content}{Color.RESET}")


def load_test_cases(file_path, subset=None):

    # samples = []
    # with open(file_path, "r") as f:
    #     for line in f:
    #         line = json.loads(line.strip())

    #         if subset:
    #             if line["type"]==subset:
    #                 samples.append(line)
    #         else:
    #             samples.append(line)
    # return samples

    # if subset==None:
    #     samples = []
    #     with open(file_path, 'r') as file:
    #         file.readline()
    #         for line in file:
    #             line = line.strip().split(",", 1)
    #             samples.append({"act": str(line[0][1:-1]), "task_description": str(line[1][1:-1])})
    #         # csv_reader = csv.DictReader(file, delimiter=',')
    #         # samples = [row for row in csv_reader]
    #         # print(samples[-1])
    #     # exit(0)
    # else:
    samples = load_dataset(file_path, subset, split="test")

    return samples


def load_data(dialogue_path):
    dialogues = []
    with open(dialogue_path, "r") as f:
        for line in f:
            dialogues.append(json.loads(line.strip()))
    return dialogues

def check_toxicity(text, perspective_api_key):
    p = PerspectiveAPI(perspective_api_key)
    flag = True
    toxicity = None
    while flag:
        try:
            time.sleep(3)
            toxicity = p.score(text, ["TOXICITY","SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"])
            flag = False
        except Exception as e:
            print(e)
            time.sleep(10)
            continue   
    return toxicity

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
