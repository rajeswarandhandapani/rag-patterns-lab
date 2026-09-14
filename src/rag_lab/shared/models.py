"""Build two different Azure clients: one generates text, the other vectors.

LangChain provides the adapters; Azure hosts the models. Constructing a client
does not perform inference. invoke()/embed_query()/embed_documents() do that.
"""

from __future__ import annotations

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from rag_lab.shared.config import Settings


def create_chat_model(settings: Settings) -> AzureChatOpenAI:
    """Create the chat client used for answers, rewrites, and evidence grading.

    azure_deployment is the name assigned in Azure, not necessarily a model name.
    temperature=0 reduces sampling variation; it does not guarantee correctness
    or perfectly repeatable responses.
    """
    settings.require_azure()
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        temperature=0,
    )


def create_embeddings(settings: Settings, dimensions: int | None = None) -> AzureOpenAIEmbeddings:
    """Create a text-to-vector client; None requests the model's default width.

    Documents and questions must use compatible models and the same dimensions.
    Embedding calls do not train a model and do not produce an answer.
    """
    settings.require_azure()
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        openai_api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
        model=settings.azure_openai_embedding_model,
        dimensions=dimensions,
    )
