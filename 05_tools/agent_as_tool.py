import asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from agents import Agent, FunctionTool, RunContextWrapper, OpenAIChatCompletionsModel, RunResult, ToolCallOutputItem,Runner,function_tool
import json
from typing_extensions import TypedDict, Any


external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",    
)

class Location(TypedDict):
    lat:float
    long:float

@function_tool
async def fetch_weather(location:Location) -> str:
    """
    fetch the weather for a given location.

    args.
    location the location to fetch the weather for.
    """
    return "sunny"

@function_tool(name_override="fetch data")
def read_file(ctx: RunContextWrapper[Any], path:str,directory:str | None = None) -> str:
    """
    read the contents of  a file.
    
    ARGS:
        path: the path to the file.
        directory: the directory to read the file from.

    """
    return "<file contents>"


agent = Agent(
        name="agent as tool",
        model=external_model,
        tools=[fetch_weather,read_file]

)   

for tool in agent.tools:
    if isinstance(tool,FunctionTool):
        print(tool.name) 
        print(tool.description) 
        print(json.dumps(tool.params_json_schema,indent=2)) 
        print()

    