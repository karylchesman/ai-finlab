from dotenv import load_dotenv
from guardrails.validators import FailResult, PassResult, register_validator
from openai import OpenAI

from guardrails import Guard, OnFailAction

load_dotenv()

client = OpenAI()


def openai_wrapper(*, messages, **kwargs) -> str:
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


@register_validator(name="simple_topic_check", data_type="string")
def simple_topic_check(value: str, metadata):
    financial_key_words = [
        "stock",
        "apple",
        "investment",
        "ticker",
        "finance",
        "market",
    ]
    if any(keyword in value.lower() for keyword in financial_key_words):
        return PassResult()
    else:
        return FailResult(
            error_message="Query is not about financial topics.",
        )


guard = Guard().use(simple_topic_check(on_fail=OnFailAction.EXCEPTION))

queries = [
    "How is Apple stock doing?",
    "What is the weather like today?",
]

for query in queries:
    print(f"Processing query: '{query}'")
    try:
        guard.validate(query)
        result = openai_wrapper(messages=[{"role": "user", "content": query}])
        print(f"Validated: {result}")
    except Exception as e:
        print(f"Blocked: {e}")
