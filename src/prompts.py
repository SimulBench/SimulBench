# encoding = "utf-8"

'''
Prompts for user agent
'''

USER_PROMPT = """Please act as a user to generate requests given the dialogue history with an AI for a given task. The task is defined in the user's first utterance. The requests should be diverse and complicated enough, and are executable commands or instructions applicable to the given task.

## Dialogue History
{DIALOGUE}

## Strategy
Some useful strategies for generating requests are:
- 1: Point out any errors or ask for further clarification.
- 2: Express your dissatisfaction and ask for improvements.
- 3: Move on to the next step of the previous requests.
- 4: Start a new request under the same task when the previous one is completed.
- 5: Others

Now, please output your strategy and a request in a JSON format by filling in the placeholders in []:
```
{
    "user": {
        "strategy": "{STRATEGY_TYPE}",
        "request": "[{REQUEST_TYPE}]"
    }
}
```
"""

'''
Prompts for evaluation
'''
EVAL_SYSTEM_RPOMPT = "You are a helpful assistant."

EVAL_PROMPT_SCORING = """[Instruction]
Please act as an impartial judge and evaluate the performance of the AI assistant in a dialogue with a user displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the AI assistant's responses to the user's requests and commands regarding the simulation task of "{SIMULATION}". Begin your evaluation by providing a short explanation. Be as objective as possible. Ignore the words of praise from the user. After providing your explanation, you must rate the response on a scale of 1 to 10 where "\n- 1 to 3: AI's responses are against the user's instructions and contain content or format errors, such as redundant explanations. \n- 4 to 6: AI's responses deviate from the user's instructions slightly but without content errors. \n- Above 7: AI's responses follow the user's instructions well and are accurate, in-depth, creative, etc.\n" and strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

[The Start of the Dialogue]
{DIALOGUE}
[The End of the Dialogue]"""

EVAL_PROMPT_SCORING_REF = """[Instruction]
Please act as an impartial judge and evaluate the performance of the AI assistant in a dialogue with a user displayed below. A full-score reference is provided for comparison. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the AI assistant's responses to the user's requests and commands regarding the simulation task of "{SIMULATION}". Begin your evaluation by providing a short explanation. Be as objective as possible. Ignore the words of praise from the user. After providing your explanation, you must rate the response on a scale of 1 to 10 where "\n- 1 to 3: AI's responses are against the user's instructions and contain content or format errors, such as redundant explanations. \n- 4 to 6: AI's responses deviate from the user's instructions slightly but without content errors. \n- Above 7: AI's responses follow the user's instructions well and are accurate, in-depth, creative, etc.\n" and strictly follow this format: \"[[rating]]\", for example: \"Rating: [[3]]\".

[The Start of the Full-score Reference]
{REFERENCE}
[The END of the Full-score Reference]

[The Start of the Dialogue]
{DIALOGUE}
[The End of the Dialogue]"""

EVAL_RPOMPT_PAIRWISE = """[Instruction]
Please act as an impartial judge and evaluate the performance of two AI assistants each in a dialogue with a user displayed below. You should choose which AI assistant performs better considering factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the AI assistant's responses to the user's requests and commands regarding the simulation task of "{SIMULATION}". The AI should adhere to the user's instructions regarding both the content and format of the responses. It should be noted that an empty response is sometimes expected in accordance with the design of a true system. Begin your evaluation by comparing the two dialogues and provide a short explanation. Avoid any position biases and ensure that the order in which the dialogues were presented does not influence your decision. Do not allow the length of the dialogues to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Ignore the words of praise from the user. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better.

[The Start of Assistant A's Dialogue]
{DIALOGUE_1}
[The End of Assistant A's Dialogue]

[The Start of Assistant B's Dialogue]
{DIALOGUE_2}
[The End of Assistant B's Dialogue]"""
