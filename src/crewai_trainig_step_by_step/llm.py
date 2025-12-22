import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

embedding_model=dict(
    provider="openai",
    config=dict(
        api_key=os.environ.get('ASIMOV_API_KEY'),
        config = dict(
            model_name = 'openai/text-embedding-3-small',
            api_key = os.environ.get('ASIMOV_API_KEY'),
            api_base = os.environ.get('ASIMOV_BASE_URL'),
            vectordb=dict(
                provider="chromadb",  # local, in-memory
            ),
        ),
    )
)

asimov_llm = LLM(
    provider="openai",
    model="openai/gpt-4o",
    api_key=os.environ.get('ASIMOV_API_KEY'),
    base_url=os.environ.get('ASIMOV_BASE_URL'),

)
