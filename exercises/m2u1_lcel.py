"""M2 U1: prompt | ChatOpenAI compatible con DeepSeek | StrOutputParser.

No utiliza una clave de OpenAI. La credencial DeepSeek se inyecta explicitamente
en memoria; no se carga .env ni se registra el cliente o sus headers.
"""
import asyncio
import os
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


async def responder(pregunta: str) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Falta DEEPSEEK_API_KEY en el entorno del proceso")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explica conceptos de Python en espanol claro, en un parrafo breve."),
        ("human", "{pregunta}"),
    ])
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=120) as sync_client:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=120) as async_client:
            model = ChatOpenAI(model="deepseek-flash", temperature=0,
                api_key=key, base_url="https://api.deepseek.com",
                http_client=sync_client, http_async_client=async_client,
                max_retries=1, max_tokens=300,
                extra_body={"thinking": {"type": "disabled"}})
            chain = prompt | model | StrOutputParser()
            output = await chain.ainvoke({"pregunta": pregunta})
    assert isinstance(output, str) and output.strip()
    return output


async def main():
    print(await responder("Por que hay que usar await con chain.ainvoke?"))


if __name__ == "__main__":
    asyncio.run(main())
