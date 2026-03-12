"""
query_optimized.py

Optimized query interface with:
  - Streaming responses
  - Chat memory support
  - Cached embeddings
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from colorama import init, Fore, Style

from core.retriever import SmartRetriever, MultiCollectionRetriever
from core.storage_manager import StorageManager
from core.source_formatter import SourceFormatter
from core.chat_memory import ChatMemoryManager

init()
logging.basicConfig(level=logging.WARNING)


def select_collection(collections):
    print("Available collections:")
    print(f"  0. {Fore.YELLOW}Search ALL collections{Style.RESET_ALL}")

    for i, name in enumerate(collections, 1):
        print(f"  {i}. {name}")

    while True:
        try:
            choice = input(f"\nSelect (0-{len(collections)}): ").strip()

            if not choice:
                continue

            n = int(choice)

            if n == 0:
                return None
            elif 1 <= n <= len(collections):
                return collections[n - 1]

        except ValueError:
            pass
        except KeyboardInterrupt:
            sys.exit(0)

        print(f"{Fore.RED}Invalid.{Style.RESET_ALL}")


def history_to_text(history):
    """Convert ChatMessage objects into readable text."""
    if not history:
        return ""

    return "\n".join(
        f"{msg.role.capitalize()}: {msg.content}"
        for msg in history
    )


def run():
    print(f"\n{Fore.CYAN}{'='*60}")
    print("  PDF QUERY — OPTIMIZED + CHAT MEMORY")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    sm = StorageManager()
    collections = sm.list_collections()

    if not collections:
        print(f"{Fore.RED}No collections. Run process_pdfs.py first.{Style.RESET_ALL}")
        return 1

    selected = select_collection(collections)

    try:
        retriever = SmartRetriever(selected) if selected else MultiCollectionRetriever()
        label = selected or "all collections"
        print(f"\n{Fore.GREEN}✓ Connected: {label}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return 1

    formatter = SourceFormatter()

    # ── Initialize Chat Memory ───────────────────────────────
    memory = ChatMemoryManager()

    print(f"{Fore.YELLOW}Type questions, 'reset' to clear memory, or 'quit' to exit{Style.RESET_ALL}\n")

    while True:
        try:
            query = input(f"{Fore.CYAN}Query: {Style.RESET_ALL}").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break

            if query.lower() == "reset":
                memory.clear()
                print(f"{Fore.YELLOW}Conversation memory cleared.{Style.RESET_ALL}\n")
                continue

            # ── Build memory context ───────────────────────────
            history = memory.get_context()
            context = history_to_text(history)

            if context:
                query_with_memory = f"""
                Previous conversation:
                {context}

                Current question:
                {query}
                """
            else:
                query_with_memory = query

            print(f"\n{Fore.GREEN}", end="", flush=True)

            if isinstance(retriever, MultiCollectionRetriever):

                resp = retriever.query_best(query_with_memory)

                if resp.retrieval_successful:
                    print(resp.answer, end="")
                    print(Style.RESET_ALL)

                    if resp.source_nodes:
                        print(formatter.format_for_terminal(resp.source_nodes))

                    memory.add_user_message(query)
                    memory.add_assistant_message(resp.answer)

                else:
                    print(f"{Fore.RED}Error: {resp.error_message}{Style.RESET_ALL}")

            else:

                result = retriever.stream(query_with_memory)

                if result.failed:
                    print(f"{Fore.RED}Error during retrieval{Style.RESET_ALL}")
                    continue

                answer_tokens = []

                for token in result:
                    answer_tokens.append(token)
                    print(token, end="", flush=True)

                print(f"{Style.RESET_ALL}\n")

                answer_text = "".join(answer_tokens)

                memory.add_user_message(query)
                memory.add_assistant_message(answer_text)

                if result.source_nodes:
                    print(formatter.format_for_terminal(result.source_nodes))

        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            break

        except Exception as e:
            print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}\n")

    return 0


if __name__ == "__main__":
    sys.exit(run())