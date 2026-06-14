from dotenv import load_dotenv
from guardrails.hub import ProfanityFree
from openai import OpenAI

from guardrails import Guard

load_dotenv()


client = OpenAI()


def openai_wrapper(*, messages, **kwargs) -> str:
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


guard = Guard().use(ProfanityFree(on_fail="exception"))
query = "FAANG representa quais fucking empresas de tecnologia?"
try:
    guard.validate(query)
except Exception as e:
    print(e)

validated_response = guard(
    openai_wrapper,
    messages=[{"role": "user", "content": query}],
)

print(validated_response.validated_output)