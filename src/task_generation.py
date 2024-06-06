# encoding = "utf-8"
from datasets import load_dataset
from random import sample
import json
import argparse
from backbone_llm import llm_generator
import os
from prompts import SYSTEM_TASK_GEN_PROMPT, TOOL_TASK_GEN_PROMPT, COMMON_PROMPT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="simulation task generation")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API_KEY")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--max_tokens", type=int, default=2048, help="The max number of tokens to be generated by the simulator")
    parser.add_argument("--temperature", type=int, default=1.2)
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--task_config_file_path", type=str, default="./data/task_specific_config.jsonl", help="path to the task specific configurations")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--new_task_count", type=int, default=10)  
    args = parser.parse_args()

    N = 5
    task_config = json.load(open(args.task_config_file_path, "r"))

    for subset in ['system', "tool"]:

        tasks = load_dataset("SimulBench/SimulBench-seed-tasks", subset, split="test")
        output_file = open(os.path.join(args.output_dir, "simulbench_"+subset+"_new.jsonl"), "a+", encoding="utf-8")
        count = 0
        
        while count < args.new_task_count:
            sampled_tasks = [tasks[id] for id in sample(list(range(0, len(tasks))), N)]
            print([sampled_tasks[i]["id"] for i in range(N)])

            sampled_tasks_strs = []
            for task in sampled_tasks:
                tmp = {}
                tmp['task_name'] = task['act']
                tmp['task_description'] = task['task_description']
                tmp['request_type'] = task_config[task["act"]]["request_type"] if (task['act'] in task_config) and ("request_type" in task_config[task["act"]]) else "your executable command or instruction"
                # sampled_tasks_strs.append("```\n" + json.dumps(tmp, indent=4) + "\n```")
                sampled_tasks_strs.append(json.dumps(tmp))

            seed_prompt = SYSTEM_TASK_GEN_PROMPT if subset == 'system' else TOOL_TASK_GEN_PROMPT
            task_gen_prompt = seed_prompt.format(*sampled_tasks_strs) + COMMON_PROMPT

            new_task = llm_generator(messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role":"user", "content":task_gen_prompt}],
                                    api_key=args.api_key,
                                    base_url=args.base_url, 
                                    temperature=args.temperature,
                                    model=args.model, 
                                    max_tokens=args.max_tokens,
                                    # is_json=False
                                    )
            
            print(new_task)
            try:
                new_task = new_task.strip()
                if new_task.startswith("```\n"):
                    new_task = new_task[4:]
                elif new_task.startswith("```json\n"):
                    new_task = new_task[8:]

                if new_task.endswith("\n```"):
                    new_task = new_task[:-4]

                new_task = json.loads(new_task)
                assert "task_name" in new_task
                assert "task_description" in new_task
                assert "request_type" in new_task
                output_file.write(json.dumps(new_task)+"\n")
                count += 1
            except Exception as e:
                print("Warning {}".format(e))
                continue

    '''
    merge data
    '''
    output_file = open(os.path.join(args.output_dir, "simulbench_all.jsonl"), "a+", encoding="utf-8")
    count = 0

    for subset in ['system', "tool"]:
        print(subset)
        tasks = load_dataset("SimulBench/SimulBench", subset, split="test")
        new_tasks = []
        with open(os.path.join(args.output_dir, "simulbench_"+subset+"_new.jsonl"), "r") as f:
            for line in f:
                new_tasks.append(json.loads(line.strip()))
        
        
        for t in tasks:
            id = str(count).zfill(3) + "#" + t["act"].replace("/", "-").replace(" ", "_")

            output_file.write(json.dumps({
                "id": id,
                "task_description": t["task_description"],
                "request_type": task_config[t["act"]]["request_type"] if (t['act'] in task_config) and ("request_type" in task_config[t["act"]]) else "your executable command or instruction",
                "task_name": t["act"],
                "type":subset
            })+"\n")

            count += 1
        
        for i,t in enumerate(new_tasks):
            print(i)
            id = str(count).zfill(3) + "#" + t["task_name"].replace("/", "-").replace(" ", "_")

            output_file.write(json.dumps({
                "id": id,
                "task_description": t["task_description"],
                "request_type": t["request_type"],
                "act": t["task_name"],
                "type": subset
            })+"\n")

            count += 1
        
