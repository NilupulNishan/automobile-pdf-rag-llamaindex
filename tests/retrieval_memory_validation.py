import sys
from pathlib import Path

# add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.retriever import SmartRetriever
from core.chat_memory import ChatMemoryManager

COLLECTION = "telecom_system_iom_procedure___starlink_system"


def run_test():

    retriever = SmartRetriever(COLLECTION)

    # initialize memory (no max_turns anymore)
    memory = ChatMemoryManager()

    print("\n=== Retrieval Accuracy Test (Memory vs No Memory) ===\n")

    # Step 1 — base question
    q1 = "What is Starlink?"

    resp1 = retriever.query(q1)

    print("Q1:", q1)
    print("Answer:", resp1.answer[:150], "...\n")

    # store conversation
    memory.add_user_message(q1)
    memory.add_assistant_message(resp1.answer)

    # Step 2 — follow-up
    q2 = "Who created it?"

    print("Follow-up:", q2)

    # WITHOUT memory
    resp_no_memory = retriever.query(q2)

    # WITH memory
    history = memory.get_context()

    context = ""
    if history:
        context = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}" for msg in history
        )

    q2_memory = f"""
Previous conversation:
{context}

Current question:
{q2}
"""

    resp_memory = retriever.query(q2_memory)

    print("\n--- Without Memory ---")
    print(resp_no_memory.answer[:150])

    print("\n--- With Memory ---")
    print(resp_memory.answer[:150])

    print("\n--- Source Comparison ---")

    print("\nWithout memory sources:")
    for node in resp_no_memory.source_nodes:
        print(node.metadata.get("page"), node.metadata.get("filename"))

    print("\nWith memory sources:")
    for node in resp_memory.source_nodes:
        print(node.metadata.get("page"), node.metadata.get("filename"))


if __name__ == "__main__":
    run_test()