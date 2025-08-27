from agents import Agent , AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig, RunContextWrapper
import os 
from dataclasses import dataclass


external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)

config = RunConfig(
    model=external_model,
    model_provider=external_client,
)

@dataclass
class Teacher:
    stu_name:str
    roll_number:int


agent = Agent(
    name="Teacher assistant",
    instructions="A helpful assistant that can answer questions and provide information.",
    model=external_model,
)


result = Runner.run_sync(agent,"nihal is a good student and his roll number is 1")


print(result.final_output)