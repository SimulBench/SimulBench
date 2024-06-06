# encoding = "utf-8"
from openai import OpenAI
from time import sleep
from gradio_client import Client
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
import torch
from fastchat.conversation import get_conv_template
import os
from together import Together
import together



def llm_generator(messages, api_key, model, base_url="https://api.openai.com/v1", max_tokens=5, temperature=1.0, is_json=False):
    
    if model in ["Qwen/Qwen1.5-7B-Chat","mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-7b-it","Qwen/Qwen1.5-110B-Chat","mistralai/Mixtral-8x22B-Instruct-v0.1","meta-llama/Llama-3-8b-chat-hf","meta-llama/Llama-3-70b-chat-hf","meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:

        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except together.error.InvalidRequestError:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                # max_tokens=max_tokens
            )
        # print(response.choices[0].message.content)
   
        return response.choices[0].message.content

    elif model in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4",
                 "gpt-4-32k", "gpt-4-turbo", "gpt-4o-2024-05-13"]:

        client = OpenAI(api_key=api_key, base_url=base_url)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,  # "gpt-3.5-turbo-16k",  # "gpt-3.5-turbo", # gpt-4-1106-preview
            temperature=temperature,
            max_tokens=max_tokens,
            # response_format = {"type":"json_object"} if is_json==True else {"type":"text"}
        )

        return chat_completion.choices[0].message.content

        '''Using hf.space as a mediator,'''
        # client = Client("https://kikiqiqi-mediator.hf.space/--replicas/xxxxx/")
        #
        # result = client.predict(
        #     json.dumps(messages),  # str  in 'Input' Textbox component
        #     json.dumps({
        #         "model": model,
        #         "api_key": api_key,
        #         "temperature": str(temperature),
        #         "base_url": base_url,
        #         "max_tokens": str(max_tokens)
        #     }),  # str  in 'Args' Textbox component
        #     api_name="/submit"
        # )
        # return result

    else:
        outputs = base_url.inference(messages=messages, temperature=temperature, max_tokens=max_tokens)
        return outputs


class ConversationPipeline:

    def __init__(self, model_dir, model_name):

        trust_remote_code = False
        if "chatglm" in model_dir.lower() or "mpt" in model_dir.lower() or "qwen" in model_dir.lower():
            trust_remote_code = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto",
                                                       trust_remote_code=trust_remote_code, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",
                                                          trust_remote_code=trust_remote_code,
                                                          torch_dtype=torch.float16).eval()
        self.conv_template = get_conv_template(model_name)
        self.model_name = model_name

    def inference(self, messages, temperature=1.0, max_tokens=100):

        if messages[0]["role"] == "system":
            self.conv_template.set_system_message(messages[0]["content"])
            messages.pop(0)

        for message in messages:
            if message["role"] == "user":
                self.conv_template.append_message(self.conv_template.roles[0], message["content"])
            elif message["role"] == "assistant":
                self.conv_template.append_message(self.conv_template.roles[1], message["content"])
            else:
                print("warning!! role unfounded")

        input_prompt = [self.conv_template.get_prompt()]
        inp = self.tokenizer(input_prompt, return_tensors="pt")

        if temperature != 0.0:
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        if self.model_name == "mistral":
            generation_config.pad_token_id = self.model.generation_config.eos_token_id
        with torch.no_grad():
            outp = self.model.generate(inp.input_ids.cuda(), generation_config=generation_config)
        output = self.tokenizer.batch_decode(outp, skip_special_tokens=True)[0]

        prompt_length = len(
            self.tokenizer.decode(
                inp.input_ids[0],
                skip_special_tokens=True,
            )
        )

        self.conv_template.messages = []

        return output[prompt_length:].strip()


if __name__ == "__main__":
    messages = [{'role': 'user', 'content': 'who is roy'}]
    api_key = ""
    print("test openai")
    results = llm_generator(messages, api_key, "gpt-3.5-turbo", max_tokens=5, temperature=1.0)
    print(results)
    # print("test llama")
    # model = LlamaPipeline("Llama-2-70b-chat-hf")s
    # results = model.inference([messages])
    # print(results)
    # print("test llamabase")
    # model = ConversationPipeline("Llama-2-7b-chat-hf", model_name="llama-2")
    # results = model.inference(messages)
    # print(results)
