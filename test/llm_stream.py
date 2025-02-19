import requests
import json

def main():
    # 1) The URL to your local LLM streaming endpoint
    #    For example, if Ollama is at http://localhost:11434 and
    #    you have an endpoint for streaming completions:
    url = "http://localhost:11434/v1/completions"  # Adjust as needed

    # 2) Example payload - you might only need a "prompt"
    payload = {
        "model": "llama3.2",
        "prompt": "Why did the chicken cross the road?",
        "stream": True
    }

    # 3) Make a streaming request
    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()

        # We'll gather all partial tokens into final_text for convenience
        final_text = []

        # 4) Iterate over lines as they arrive
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            # The line might look like:
            # data: {"id":"cmpl-746","object":"text_completion","created":1740004115, ... }
            # Remove the 'data: ' prefix if present
            line = raw_line.removeprefix("data: ")
            if line.strip() == "[DONE]":
                # Some servers send a sentinel "[DONE]" line at the end
                break

            # 5) Parse the JSON portion
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # If it's not valid JSON, skip or handle differently
                continue

            # We assume something like:
            # {"choices":[{"text":" partial token","index":0,"finish_reason":null}],...}
            choices = data.get("choices", [])
            if not choices:
                continue

            # Extract the partial text
            partial_text = choices[0].get("text", "")
            # Append to our final_text list
            final_text.append(partial_text)

            # Print partial token to console (no newline)
            print(partial_text, end="", flush=True)

        # 6) Done streaming. Print a newline
        print("\n\n--- Streaming Complete ---\n")

        # If you want the entire combined text:
        combined = "".join(final_text)
        print("Final text:", combined)

if __name__ == "__main__":
    main()
