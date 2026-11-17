"""Interactive Phase 1 Text Completion Tester for OpenMythos 1B."""

from hive_common import run_interactive_tester

if __name__ == "__main__":
    run_interactive_tester(
        title="OpenMythos 1B — Phase 1 Completion Tester",
        max_new_tokens=50,
        temperature=0.8,
        print_prompt=True,
    )
