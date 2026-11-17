"""Interactive Phase 2 Conversational Chat Tester for OpenMythos 1B."""

import os

from hive_common import load_dotenv, run_interactive_tester

load_dotenv()

SYSTEM_PROMPT = os.environ.get(
    "HIVE_SYSTEM_PROMPT",
    "You are a helpful, honest AI companion. Think carefully and converse naturally.",
)

if __name__ == "__main__":
    run_interactive_tester(
        title="OpenMythos 1B — Phase 2 Chat Tester",
        prompt_formatter=lambda msg: f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{msg}\n<|ai|>\n",
        stop_tokens=("<|endoftext|>", "<|user|>"),
        max_new_tokens=60,
        temperature=0.7,
        print_prompt=False,
    )
