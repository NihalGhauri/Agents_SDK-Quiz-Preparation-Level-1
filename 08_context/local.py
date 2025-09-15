from agents import Agent , Runner , AsyncOpenAI, OpenAIChatCompletionsModel,RunContextWrapper,function_tool
import os
import asyncio
from dataclasses import dataclass

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)


@dataclass
class UserInfo:
    name:str
    uid:int

@function_tool
def get_user_name(wrapper:RunContextWrapper[UserInfo])-> str:
    # """Fetch the age and name of the user. Call this function to get user's name and age information."""
    return f"The user name is {wrapper.context.name} and his is 20 years old"

@function_tool
def get_uid(wrapper:RunContextWrapper[UserInfo])-> str:
    # """Fetch the user_id of the user. Call this function to get user's id information."""
    return f"The user id is {wrapper.context.uid}"

async def main():
    user_info = UserInfo(
        name="nihal khan ghauri",
        uid=5
    )

    agent = Agent[UserInfo](
        name = "assistant",
        model=external_model,
        tools=[get_user_name,get_uid]

    )

    result = await Runner.run(
        starting_agent=agent,
         input= "what is the user name and his age?",
         context=user_info)
    
    print(result.final_output)


asyncio.run(main())
