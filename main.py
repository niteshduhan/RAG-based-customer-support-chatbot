from generator import answer

def main():
    print("🤖 Amazon Customer Service Agent")
    print("   Type 'exit' to quit\n")

    history = []

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # answer() returns (text, history, meta) — unpack all three
        response, history, _ = answer(query, history)
        print(f"\nAgent: {response}\n")
        print("─" * 60)

if __name__ == "__main__":
    main()