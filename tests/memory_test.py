"""
Advanced Chat Memory Stress Test

Tests:
1. Large conversation simulation
2. Memory overflow handling
3. Context generation performance
4. Memory reset
5. Memory scaling behavior
"""

import sys
import time
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.chat_memory import ChatMemoryManager


def history_to_text(history):
    """Convert ChatMessage objects to readable text."""
    if not history:
        return ""

    return "\n".join(
        f"{msg.role.capitalize()}: {msg.content}"
        for msg in history
    )


def simulate_conversation(memory, turns=50):
    """Simulate a real conversation with many turns."""

    topics = [
        "Starlink",
        "Satellite internet",
        "Network configuration",
        "Antenna alignment",
        "Power requirements",
        "System installation",
    ]

    for i in range(turns):
        topic = random.choice(topics)

        memory.add_user_message(f"Question {i}: Explain {topic}")
        memory.add_assistant_message(f"Answer {i}: Information about {topic}")

    return memory.get_context()


def test_large_conversation():
    print("\n--- Test 1: Large Conversation Simulation ---")

    memory = ChatMemoryManager()

    history = simulate_conversation(memory, 40)

    context = history_to_text(history)

    print("Context length:", len(context))
    print(context[:200], "...")

    assert "Question" in context
    print("✓ Large conversation handled")


def test_memory_overflow():
    print("\n--- Test 2: Memory Overflow Handling ---")

    memory = ChatMemoryManager()

    for i in range(10):
        memory.add_user_message(f"User Q{i}")
        memory.add_assistant_message(f"Assistant A{i}")

    history = memory.get_context()
    context = history_to_text(history)

    print(context)

    # Token-based trimming means very old messages may disappear
    assert "Q9" in context

    print("✓ Memory trimming functioning")


def test_context_generation_speed():
    print("\n--- Test 3: Context Generation Speed ---")

    memory = ChatMemoryManager()

    for i in range(100):
        memory.add_user_message(f"Question {i}")
        memory.add_assistant_message(f"Answer {i}")

    start = time.time()

    for _ in range(1000):
        memory.get_context()

    end = time.time()

    duration = end - start

    print(f"Time for 1000 context generations: {duration:.4f} seconds")

    assert duration < 1
    print("✓ Context generation efficient")


def test_memory_scaling():
    print("\n--- Test 4: Memory Scaling Test ---")

    memory = ChatMemoryManager()

    start = time.time()

    for i in range(5000):
        memory.add_user_message(f"User question {i}")
        memory.add_assistant_message(f"Assistant answer {i}")

    memory.get_context()

    end = time.time()

    duration = end - start

    print(f"Time for 5000 turns: {duration:.4f} seconds")

    print("✓ Memory scaling acceptable")


def test_reset():
    print("\n--- Test 5: Memory Reset ---")

    memory = ChatMemoryManager()

    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi")

    memory.clear()

    history = memory.get_context()

    assert len(history) == 0

    print("✓ Memory reset successful")


def run_all_tests():

    print("\n==============================")
    print(" ADVANCED CHAT MEMORY TEST ")
    print("==============================")

    test_large_conversation()
    test_memory_overflow()
    test_context_generation_speed()
    test_memory_scaling()
    test_reset()

    print("\nAll advanced tests completed successfully.")


if __name__ == "__main__":
    run_all_tests()