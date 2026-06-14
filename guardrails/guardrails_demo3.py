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
def openai_topic_check(value: str, metadata):
    prompt = f"""
    Is this query about financial analysis or stocks? Answer with a simple YES or NO.
    Query: {value}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip().upper()
    if answer == "YES":
        return PassResult()
    else:
        return FailResult(
            error_message="Query is not about financial topics.",
        )


guard = Guard().use(openai_topic_check(on_fail=OnFailAction.EXCEPTION))

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

# OWASP Top 10 for LLM Application: to know more about the main security risks when building LLM applications
