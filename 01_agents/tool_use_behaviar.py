# from agents import (
#     Agent,
#     AsyncOpenAI,
#     OpenAIChatCompletionsModel,
#     Runner,
#     RunConfig,
#     RunContextWrapper,
#     function_tool,
#     ModelSettings
# )
# import os


# external_client = AsyncOpenAI(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# external_model = OpenAIChatCompletionsModel(
#     openai_client=external_client,
#     model="gemini-1.5-flash",
# )

# config = RunConfig(
#     model=external_model,
#     model_provider=external_client,
# )

# @function_tool
# def get_weather(city: str) -> str:
#     """Returns weather info for the specified city."""
#     return f"The weather in {city} is sunny"

# # agent = Agent(
# #     name="Weather Agent",
# #     instructions="Retrieve weather details.",
# #     tools=[get_weather],
# #     model_settings=ModelSettings(tool_choice="get_weather"),
# #     model=external_model,
# # )


# agent_1 = Agent(
#     name="Weather Agent",
#     instructions="Retrieve weather details.",
#     tools=[get_weather],
#     tool_use_behavior="stop_on_first_tool",
#     model=external_model,
# )

# # result = Runner.run_sync(agent, "What is the weather in New York?")
# result_1 = Runner.run_sync(agent_1, "What is the weather in karachi?")

# print(f"agent_1       ===>   {result_1.final_output}")
# # print(result.final_output)
# # print(result.last_agent.name)


from agents import Agent, AsyncOpenAI,OpenAIChatCompletionsModel,Runner, set_tracing_disabled , function_tool
import os
from agents.agent import StopAtTools

set_tracing_disabled(True)


external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",

)



@function_tool
def get_weather(city:str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is rainy "




@function_tool
def sum_nums(a:int , b:int) -> int:
    """" add two numbers"""
    return a + b * 2


agent = Agent(
    name="stop at stock agents",
    instructions="get weather or sum numbers",
    model=external_model,
    tools=[get_weather, sum_nums],
    tool_use_behavior=StopAtTools(
        stop_at_tools_names=["get_weather"]
        )
)

result = Runner.run_sync(agent, "2+2")
print(result.final_output)
print(result.last_agent.name)
