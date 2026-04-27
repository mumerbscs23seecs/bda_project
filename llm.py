from openai import OpenAI

def get_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )


def generate_answer(client, model, query, context_chunks):

    context = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
You are a QA system.

Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content