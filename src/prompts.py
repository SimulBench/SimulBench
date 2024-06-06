# encoding = "utf-8"

'''
Prompts for new challenging task generation using 5-shot demonstration
'''

SYSTEM_TASK_GEN_PROMPT = """Please act as a task creator and it is your job to create challenging tasks that require **expertise** of diverse domains. Each task is defined as an interface that, at each turn, it accepts an user request(textual and self-contained) as input and returns the output by following the underlying mechanism or rules of the task.
It is very important that the tasks you create should have a real underlying **environment/state** that **evolves** as user's request is processed at each turn. Each task is composed of the following four attributes: 
- task_name: a short noun phrase representing the name of the task
- task_description: a paragraph describing the task. It should contain information about (1) a short description of the task and the expected general form of input; (2) the expected general form of output, and some additional style/format constraints regarding the output; (3) the first query provided by the real user. Should start with 'My/The first ...' and contain a concrete query
- request_type: the general form of each user request

Below are some exemplars regarding different possible tasks, for your reference:
{}
{}
{}
{}
{}
"""

TOOL_TASK_GEN_PROMPT = """Please act as a task creator and it is your job to create challenging and diverse tasks that require **expertise** of diverse domains. Each task is defined as a practical **stateless** one-time call interface that accept an user request(textual and self-contained) as input and return the output by following the underlying mechanism or rules of the task.
Please note that the tasks you create should not be without evaluation criteria and should not be entirely creative. Each task is composed of the following four attributes: 
- task_name: a short noun phrase representing the name of the task
- task_description: a paragraph describing the task. It should contain information about (1) a short description of the task and the expected general form of input; (2) the expected general form of output, and some additional style/format constraints regarding the output; (3) the first query provided by the real user. Should start with 'My/The first ...' and contain a concrete query
- request_type: the general form of each user query

Below are some exemplars regarding different possible tasks, for your reference:
{}
{}
{}
{}
{}
"""

COMMON_PROMPT = """
Now, please come up with a new **challenging** and **diverse** task and output it in JSON format by filling the placeholders in [] in the follolwing code block. Only output the JSON code block, nothing else!
```
{
    "task_name": [TASK_NAME],
    "task_description": [TASK_DESCRIPTION],
    "request_type": [REQUEST_TYPE]
}
```
"""

'''
Prompts for user agent
'''

USER_PROMPT = """Please act as a user who is fond of posing requests, given the interaction history between the user and an AI assistant for a given task. The name and description of the task are provided in the user's **first** utterance. The requests you generate should be **diverse** and **complicated** enough and be executable commands or instructions applicable to the given task.

## Interaction History
{DIALOGUE}

## Strategy
Here are some useful strategies for generating the next request:
- 1: A request that points out any errors, ambiguities, or other **dissatisfactions** of the previous response from the AI assistant.
- 2: A request that never appeared previously but is **related** to previous requests by either conditioning on previous requests' outcome or will have an impact upon previous outcomes.
- 3: A request that never appeared previously, still belongs to the same domain as previous requests but has a much higher **rarity** (i.e., being more long-tailed), or a much higher **difficulty** and **complexity**.
- 4: Others

Now, please first select one strategy from the above options (if Others is selected, please further elaborate on your strategy) and then generate a new request using the selected strategy. Put the strategy number and request in JSON format by filling in the placeholders in [] in the following code block. Only output the JSON code block, nothing else!
```
{
    "user": {
        "strategy": "[{STRATEGY_TYPE}]",
        "request": "[{REQUEST_TYPE}]"
    }
}
```
"""

'''
Prompts for evaluation
'''
EVAL_SYSTEM_RPOMPT = "You are a helpful assistant."

EVAL_DIFFICULT_TURN_PROMPT = """[Instruction]
Please act as a keen observer and a sharp-eyed judge. You will be presented with a multi-turn dialogue history between a user and an AI assistant. In the dialogue, the user may pose various types of requests to the AI assistant and the AI assistant should (but may fail to) provide relevant and accurate response.
Your job is to carefully look through each turn in the dialogue and identify the first **challenging** turn in the dialogue. Challenging implies that, at that turn, the user's request possesses extreme complexity and difficulty, resulting in an inaccurate response(content errors, format errors, ambiguities, etc.) from the AI assistant.

[The Start of the Dialogue]
{DIALOGUE}
[The End of the Dialogue]

Now, please carefully look through each turn in the above dialogue and identify the first **challenging** turn in it. Your decision should have an objective and rigorous rationale to support it. Begin your evaluation by providing a short explanation. After providing the explanation, you must output the turn number(1 to {DIALOGUE_LEN}, and 0 represents no challenging turns), and strictly follow this format: \"[[TURN_NUMBER]]\", for example: \"Turn: [[3]]\".
"""

EVAL_PROMPT_SCORING = """[Instruction]
Please act as an impartial and sharp-eyed judge. You will be presented with a multi-turn dialogue history between a user and an AI assistant. In the dialogue, the user may pose various types of requests to the AI assistant, and the AI assistant should (but may fail to) provide a high-quality response to satisfy the user's need. 
Your job is to evaluate the quality of the **last response** provided by the AI assistant based on the earlier dialogue history. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the AI assistant's responses to the user's requests and commands regarding the simulation task of "{SIMULATION}".

Begin your evaluation by providing a short explanation. Be as objective as possible. Ignore the words of praise from the user. After providing your explanation, you must rate the response on a scale of 1 to 10. The scoring standard is as follows:
- 1 to 2: The AI's response terribly fulfills the user's request, completely deviating from its simulation task and containing garbage content and format errors.
- 3 to 4: The AI's response poorly fulfills the user's request, containing inaccurate and useless content given its simulation task, or having format errors such as redundant explanations.
- 5 to 6: The AI's response moderately fulfills the user's request, containing no content errors but still showing errors such as incorrect format and repetition.
- 7 to 8: The AI's response well fulfills the user's request, containing no content errors and strictly following format requirements.
- 9 to 10: The AI's response perfectly fulfills the user's request, containing no content or format errors at all while exhibiting extremely good relevance, helpfulness, accuracy, and creativity. 
Your rating should strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

[The Start of the Dialogue History]
{DIALOGUE}
[The End of the Dialogue History]

[The Start of the Last Response]
{RESPONSE}
[The End of the Last Response]
"""


EVAL_RPOMPT_PAIRWISE = """[Instruction]
Please act as an impartial and sharp-eyed judge. You will be presented with a multi-turn dialogue history between a user and an AI assistant, and two candidates of the **next response** from the AI assistant. In the dialogue, the user may pose various types of requests to the AI assistant, and the AI assistant should provide a high-quality response to satisfy the user's need. 
Your job is to evaluate which candidate response is better considering factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the AI assistant's responses to the user's requests and commands regarding the simulation task of "{SIMULATION}". The AI should adhere to the user's instructions regarding both the content and format of the responses. Additional explanations not required unless been asked, and should be punished if they are unallowed by the simulation task. It should be noted that an empty response is sometimes expected in accordance with the design of a true system. 

Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain indexes of the responses. Be as objective as possible. Ignore the words of praise from the user. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if response A is better, \"[[B]]\" if response B is better, and \"[[C]]\" for a tie.

[The Start of the Dialogue History]
{DIALOGUE}
[The End of the Dialogue History]

[The Start of Candidate Response A]
{RESPONSE_1}
[The End of Candidate Response A]

[The Start of Candidate Response B]
{RESPONSE_2}
[The End of Candidate Response B]"""


