"""Example usage of LazyRouter with OpenAI SDK"""

from openai import OpenAI


def main():
    # Create client pointing to LazyRouter5
    client = OpenAI(
        base_url="http://localhost:1234/v1", api_key="dummy"  # Not used, but required by SDK
    )

    print("=" * 60)
    print("LazyRouter Example Usage")
    print("=" * 60)

    # Example 1: Simple query (should route to cheap model)
    print("\n1. Simple Query (automatic routing):")
    print("   Query: 'What is 2+2?'")
    response = client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    print(f"   Model used: {response.model}")
    print(f"   Response: {response.choices[0].message.content}")

    # Example 2: Complex query (should route to powerful model)
    print("\n2. Complex Query (automatic routing):")
    print("   Query: 'Explain quantum entanglement...'")
    response = client.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "user",
                "content": "Explain quantum entanglement and its implications for quantum computing",
            }
        ],
    )
    print(f"   Model used: {response.model}")
    print(f"   Response: {response.choices[0].message.content[:200]}...")

    # Example 3: Coding task
    print("\n3. Coding Task (automatic routing):")
    print("   Query: 'Write a Python function...'")
    response = client.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function to calculate the nth Fibonacci number",
            }
        ],
    )
    print(f"   Model used: {response.model}")
    print(f"   Response: {response.choices[0].message.content[:200]}...")

    # Example 4: Manual model selection
    print("\n4. Manual Model Selection:")
    print("   Forcing model: 'gemini-3-pro-preview'")
    response = client.chat.completions.create(
        model="gemini-3-pro-preview",
        messages=[{"role": "user", "content": "Hello!"}],  # Bypass router
    )
    print(f"   Model used: {response.model}")
    print(f"   Response: {response.choices[0].message.content}")

    # Example 5: Streaming
    print("\n5. Streaming Response:")
    print("   Query: 'Count from 1 to 5'")
    print("   Response: ", end="", flush=True)
    stream = client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "Count from 1 to 5"}], stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure LazyRouter is running:")
        print("  python main.py")
