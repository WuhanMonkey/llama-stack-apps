# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import fire
import os
import json
import uuid
import pandas as pd
from tqdm import tqdm

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from termcolor import cprint
from .util import data_url_from_file


def build_index(client: LlamaStackClient, file_dir: str, bank_id: str) -> str:
    """Build a memory bank from a directory of pdf files"""
    # 1. create memory bank
    providers = client.providers.list()
    client.memory_banks.register(
        memory_bank={
            "identifier": bank_id,
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
            "provider_id": providers["memory"][0].provider_id,
        }
    )

    # 2. load pdfs from directory as raw text
    paths = []
    for filename in os.listdir(file_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(file_dir, filename)
            paths.append(file_path)

    documents = [
        Document(
            document_id=os.path.basename(path),
            content=data_url_from_file(path),
            mime_type="application/pdf",
        )
        for path in paths
    ]

    # insert some documents
    client.memory.insert(bank_id=bank_id, documents=documents)

    return bank_id

async def get_response_row(agent: Agent, input_query: str) -> str:
    # single turn, each prompt is a new session
    session_id = agent.create_session(f"session-{input_query}")
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": input_query,
            }
        ],
        session_id=session_id,
    )

    async for chunk in response:
        event = chunk.event
        event_type = event.payload.event_type
        if event_type == "turn_complete":
            return event.payload.turn.output_message.content


async def run_main(host: str, port: int, docs_dir: str, input_file_path: str):
    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    bank_id = "rag_agent_docs"
    build_index(client, docs_dir, bank_id)
    print(f"Created bank: {bank_id}")

    agent_config = AgentConfig(
        model="Llama3.1-405B-Instruct",
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": [{"bank_id": bank_id, "type": "vector"}],
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 4096,
                "max_chunks": 10,
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config)

    # load dataset and generate responses for the RAG agent
    df = pd.read_csv(input_file_path)
    user_prompts = df["input_query"].tolist()

    llamastack_generated_responses = []

    for prompt in tqdm(user_prompts):
        print(f"Generating response for: {prompt}")
        try:
            generated_response = await get_response_row(agent, prompt)
            llamastack_generated_responses.append(generated_response)
        except Exception as e:
            print(f"Error generating response for {prompt}: {e}")
            llamastack_generated_responses.append(None)

    df["generated_answer"] = llamastack_generated_responses

    output_file_path = input_file_path.replace(".csv", "_llamastack_generated.csv")
    df.to_csv(output_file_path, index=False)
    print(f"Saved to {output_file_path}")


def main(host: str, port: int, docs_dir: str, input_file_path: str):
    asyncio.run(run_main(host, port, docs_dir, input_file_path))


if __name__ == "__main__":
    fire.Fire(main)
