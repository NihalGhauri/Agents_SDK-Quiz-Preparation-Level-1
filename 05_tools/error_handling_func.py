# from agents import Agent , Runner , OpenAIChatCompletionsModel, AsyncOpenAI,RunConfig,function_tool,RunContextWrapper
# from typing import Any
# import os 

# external_client = AsyncOpenAI(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# external_model= OpenAIChatCompletionsModel(
#     openai_client=external_client,
#     model="gemini-2.0-flash"
# )

# def my_custom_error_function(ctx:RunContextWrapper[Any],error:Exception)->str:
#     print(f"a tool call failed with the following error: {error}")
#     return "an internal server error occured.  please try again later"


# @function_tool(failure_error_function=my_custom_error_function)
# def get_user_profile(user_id:str) -> str:
#     id user_id == ""