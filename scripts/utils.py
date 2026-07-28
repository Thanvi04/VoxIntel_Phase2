import re

# Remove hidden Unicode characters
def clean_unicode(text):

    text = str(text)

    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    return text.strip()


# Remove multiple spaces
def remove_extra_spaces(text):

    text = str(text)

    return re.sub(r"\s+", " ", text).strip()


# Remove spaces around commas
def normalize_commas(text):

    text = str(text)

    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)

    return text.strip()


# Check for English letters
def contains_english(text):

    return bool(re.search(r"[A-Za-z]", str(text)))


# Check for digits
def contains_digits(text):

    return bool(re.search(r"\d", str(text)))