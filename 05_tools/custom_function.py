from typing import Any
from pydantic import BaseModel
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig,
    function_tool,
    FunctionTool,
)
import os

external_client= AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)

def do_some_work(data:str)-> str:
    return f"Processed: {data}"


class FunctionArgs(BaseModel):
    username:str
    age:int


async def run_function(ctx:Any,args:str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return do_some_work(data=f"{parsed.username} is {parsed.age} years old and he is driving a corrolla")


tool = FunctionTool(
     name="process_user",
     description="processs extracted user data",
     params_json_schema=FunctionArgs.model_json_schema(),
     on_invoke_tool=run_function,
)

agent = Agent(
    name="assistant",
    instructions="you are a helpful assistant that can proceed user data and answer questions",
    model=external_model,
    tools=[tool],
)

result = Runner.run_sync(
    agent, 'Process user data: {"username": "nihal khan ghauri", "age": 20}'
)

print(result.final_output)
