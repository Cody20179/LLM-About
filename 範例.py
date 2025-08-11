from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, ModelSettings, function_tool
import matplotlib.pyplot as plt
import numpy as np
import os

with open("Prompt/Openaikey.txt", 'r', encoding="utf-8") as f:
    content = f.read()
    os.environ["OPENAI_API_KEY"] = content

external_client = AsyncOpenAI(
    base_url="http://192.168.0.150:11434/v1", # "http://localhost:11434/v1"
    api_key="ollama",
)

@function_tool
def Tool_Example(Num:int):
     print("Call Tool Example")
     Data = np.linspace(0, Num, Num)
     return Data.tolist()

@function_tool
def Trun_to_Sin(Data:list):
    print("Call Tool Trun to sin")
    SinData = np.sin(Data)
    return SinData.tolist()

@function_tool
def Plot(Data:list):
    print("Call Tool Plot")
    plt.figure(figsize=(6, 3))
    plt.plot(Data)
    plt.show()
    return "Plot is finish"

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model=OpenAIChatCompletionsModel(
        model="qwen3:0.6b",
        openai_client=external_client
    ),
    model_settings=ModelSettings(
        temperature=0.2,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3
    ),
    tools=[Tool_Example, Trun_to_Sin, Plot], # PostgreSQL
)

async def main(user_prompt):
    print(user_prompt)
    background_prompt = """
    You are a LLM Assister, 
    You Must help user to finish work when they ask,
    If User ask you to plot a sin wave figure,
    you must call function 1. Tool_Example -> 2. Trun_to_Sin -> 3. Plot
    and return to user
""" 
    result = await Runner.run(
        starting_agent = agent, 
        input = background_prompt + user_prompt,
        max_turns = 8
        )
    print(f"User's question:{user_prompt}")
    print(f"system Results:{result.final_output}")

if __name__ == "__main__":
    user_prompt = "Help me to plot a figure with 10 sin Data "
    await main(user_prompt)