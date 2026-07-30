from transformers import AutoTokenizer

# ============================================================
# MODELS
# ============================================================

MODELS = {
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Gemma-3-4B-IT": "google/gemma-3-4b-it",
}


# ============================================================
# SAME TULU SENTENCES FOR BOTH MODELS
# ============================================================

test_sentences = [
    "ನೀರ್",
    "ಮನೆ",
    "ಎಂಚ ಉಲ್ಲರ್?",
    "ಎಂಕ್ ಎಡ್ಡೆ ಉಲ್ಲೆ, ಸೊಲ್ಮೆಲು.",
    "ಈರ್ನ ಪುದರ್ ದಾದ?",
    "ಯಾನ್ ನಿಕ್ಲೆಗ್ ಸಹಾಯ ಮಲ್ತೊಂದುಲ್ಲೆ."
]


# ============================================================
# STORE RESULTS
# ============================================================

results = {}


print("=" * 70)
print("TULU TOKENIZER COMPARISON")
print("=" * 70)


# ============================================================
# TEST EACH MODEL
# ============================================================

for model_name, model_id in MODELS.items():

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("MODEL ID:", model_id)
    print("=" * 70)

    try:

        print("\nLoading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Tokenizer loaded successfully.\n")

        total_tokens = 0
        total_characters = 0

        for text in test_sentences:

            encoded = tokenizer(
                text,
                add_special_tokens=False
            )

            token_ids = encoded["input_ids"]

            token_count = len(token_ids)
            character_count = len(text)

            total_tokens += token_count
            total_characters += character_count

            print("Text       :", text)
            print("Characters :", character_count)
            print("Tokens     :", token_count)

            print("-" * 50)


        average_tokens = total_tokens / len(test_sentences)

        characters_per_token = (
            total_characters / total_tokens
            if total_tokens > 0
            else 0
        )


        results[model_name] = {
            "status": "SUCCESS",
            "total_characters": total_characters,
            "total_tokens": total_tokens,
            "average_tokens": average_tokens,
            "characters_per_token": characters_per_token
        }


    except Exception as error:

        print("\nCould not load tokenizer.")
        print("Error:")
        print(error)

        results[model_name] = {
            "status": "FAILED",
            "error": str(error)
        }


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL TULU TOKENIZER COMPARISON")
print("=" * 70)


successful_models = []


for model_name, result in results.items():

    print("\nMODEL:", model_name)

    if result["status"] == "SUCCESS":

        print(
            "Total characters       :",
            result["total_characters"]
        )

        print(
            "Total tokens           :",
            result["total_tokens"]
        )

        print(
            "Average tokens/sentence:",
            round(result["average_tokens"], 2)
        )

        print(
            "Characters per token   :",
            round(result["characters_per_token"], 2)
        )

        successful_models.append(
            (
                model_name,
                result["total_tokens"]
            )
        )

    else:

        print("Status: FAILED")


# ============================================================
# FIND MOST TOKEN-EFFICIENT MODEL
# ============================================================

if successful_models:

    successful_models.sort(
        key=lambda item: item[1]
    )

    best_model = successful_models[0][0]

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "Most token-efficient model for these Tulu samples:",
        best_model
    )

    print(
        "\nNOTE: Tokenizer efficiency is only one factor "
        "when selecting the final model."
    )

else:

    print("\nNo tokenizer could be tested successfully.")


print("\nTokenizer comparison completed.")