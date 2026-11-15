"""STEP 5 - GENERATION

Put the re-ranked chunks into a prompt and ask an LLM (an OpenAI model
deployed in Azure AI Foundry) to answer using ONLY that context.
If Azure is not configured, return the prompt so the demo still works offline.
"""
from backend import config

SYSTEM_PROMPT = (
    "You are a helpful customer-support assistant for an online store. "
    "Answer the question using ONLY the context below. Cite the chunk numbers you used, "
    "like [1] or [2]. If the context does not contain the answer, say you don't know. "
    "Reply in plain text without markdown."
)


DOCUMENT_SYSTEM_PROMPT = (
    "You answer questions about a document the user uploaded. "
    "Answer using ONLY the context below. Cite the chunk numbers you used, like [1] or [2]. "
    "If the context does not contain the answer, say you don't know. "
    "Reply in plain text without markdown."
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n".join(f"[{i}] ({c['chunk']['source']}) {c['chunk']['text']}"
                        for i, c in enumerate(chunks, start=1))
    return f"Context:\n{context}\n\nQuestion: {question}"


def azure_configured() -> bool:
    return all([config.AZURE_OPENAI_ENDPOINT, config.AZURE_OPENAI_API_KEY, config.AZURE_OPENAI_DEPLOYMENT])


def generate(question: str, chunks: list[dict], system_prompt: str = SYSTEM_PROMPT) -> dict:
    prompt = build_prompt(question, chunks)
    out = {"system_prompt": system_prompt, "prompt": prompt, "answer": None,
           "llm": None, "error": None}
    if not azure_configured():
        out["error"] = ("Azure OpenAI is not configured (see .env.example). "
                        "Showing the prompt that would be sent to the LLM.")
        return out
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                             api_key=config.AZURE_OPENAI_API_KEY,
                             api_version=config.AZURE_OPENAI_API_VERSION)
        resp = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,   # the deployment name in Foundry
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
        )
        out["answer"] = resp.choices[0].message.content
        out["llm"] = f"Azure OpenAI / {config.AZURE_OPENAI_DEPLOYMENT}"
    except Exception as e:  # show the error in the UI instead of crashing the demo
        out["error"] = f"Azure OpenAI call failed: {e}"
    return out
