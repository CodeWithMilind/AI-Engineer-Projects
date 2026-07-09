from services.ollama_service import OllamaClient, OllamaError


def main() -> None:
    client = OllamaClient(timeout=120, max_tokens=50)
    prompt = "What is machine learning?"

    try:
        response = client.generate(prompt)
        print("Ollama response:")
        print(response)
    except OllamaError as exc:
        print(f"Ollama test failed: {exc}")
    except Exception as exc:
        print(f"Unexpected error during Ollama test: {exc}")


if __name__ == "__main__":
    main()
