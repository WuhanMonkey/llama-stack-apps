# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from llama_stack_client.types.agent_create_params import AgentConfig

# NOTE: We may need to modify the prompt to explicitly tell the agent to use the search tool.

AGENT_CONFIG = AgentConfig(
    model="Llama3.1-405B-Instruct",
    instructions="You are a helpful assistant",
    sampling_params={
        "strategy": "greedy",
        "temperature": 1.0,
        "top_p": 0.9,
    },
    tools=[
        {
            "type": "brave_search",
            "engine": "brave",
            "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
        }
    ],
    tool_choice="auto",
    tool_prompt_format="json",
    input_shields=[],
    output_shields=[],
    enable_session_persistence=False,
)
