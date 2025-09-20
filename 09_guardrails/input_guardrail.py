from pydantic import BaseModel
from agents import Agent,AsyncOpenAI, Runner, OpenAIChatCompletionsModel,input_guardrail,RunContextWrapper ,GuardrailFunctionOutput,InputGuardrailTripwireTriggered,TResponseInputItem,enable_verbose_stdout_logging
import asyncio
import os

enable_verbose_stdout_logging()

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",

)

external_model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=external_client
    )

class MathHomeWorkOutput(BaseModel):
    is_math_home_work : bool
    reasoning : str


guardrail_agent = Agent(
    name="guardrail check",
    instructions="check if the user is asking you to do their math homework",
    output_type = MathHomeWorkOutput,
    model=external_model
)

@input_guardrail
async def math_guardrail(
    ctx: RunContextWrapper[None] ,agent: Agent, input: str | list[TResponseInputItem]

)-> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input,context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math_home_work
    )

agent = Agent(
    name="Customer support agent ",
    instructions="you are a customer support agent. you help customers with their questions ",
    input_guardrails=[math_guardrail],
    model=external_model
)

async def main():
    try:
        result = await Runner.run(agent, "2+2")
        print(result.final_output)
        print("Guardrail did not trip this is unexpected")

    except InputGuardrailTripwireTriggered:
        print("Math home work guardrail tripped")

asyncio.run(main())
