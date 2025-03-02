import json
import tiktoken
import re

# Define file paths
input_file = "fine_tuning_data.jsonl"
output_file = "edit_fine_tuning_data.jsonl"

# Set the token limit
TOKEN_LIMIT = 333333
MODEL_NAME = "gpt-4o-mini-2024-07-18"

# Initialize tokenizer
encoding = tiktoken.encoding_for_model(MODEL_NAME)

# Regex pattern to detect URLs
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

def count_tokens(text):
    """Returns the number of tokens in a given text."""
    return len(encoding.encode(text))

# Read all data from the input file
data_list = []
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)

        # Count total tokens excluding messages with links
        total_tokens = sum(count_tokens(msg["content"]) for msg in data["messages"] if not URL_PATTERN.search(msg["content"]))

        # Set weight to 1 for all assistant messages
        for msg in data["messages"]:
            if msg.get("role") == "assistant":
                msg["weight"] = 1

        # Store data along with its token count
        data_list.append((total_tokens, data))

# Sort examples by token count (largest first)
data_list.sort(reverse=True, key=lambda x: x[0])

# Select messages while staying under the token limit
selected_data = []
current_token_count = 0

for tokens, data in data_list:
    # Count only messages without links
    valid_tokens = sum(count_tokens(msg["content"]) for msg in data["messages"] if not URL_PATTERN.search(msg["content"]))

    if current_token_count + valid_tokens > TOKEN_LIMIT:
        break

    selected_data.append(data)
    current_token_count += valid_tokens

# Write the filtered data back to a new file
with open(output_file, "w", encoding="utf-8") as outfile:
    for data in selected_data:
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Processed file saved as {output_file}, with a total of {current_token_count} tokens (excluding link messages).")
