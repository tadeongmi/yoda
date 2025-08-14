"""Yoda speech transformation logic."""

import re


class YodaTransformer:
    """Transform text to mimic Yoda's speech patterns."""

    def __init__(self) -> None:
        """Initialize the transformer with speech patterns."""
        # Common Yoda patterns: move objects/predicates to beginning
        self.patterns = [
            # "You are X" -> "X you are"
            (r"^you are (.+)$", r"\1 you are"),
            # "I am X" -> "X I am"
            (r"^i am (.+)$", r"\1 I am"),
            # "You will X" -> "X you will"
            (r"^you will (.+)$", r"\1 you will"),
            # "I will X" -> "X I will"
            (r"^i will (.+)$", r"\1 I will"),
            # "You have X" -> "X you have"
            (r"^you have (.+)$", r"\1 you have"),
            # "I have X" -> "X I have"
            (r"^i have (.+)$", r"\1 I have"),
            # "You can X" -> "X you can"
            (r"^you can (.+)$", r"\1 you can"),
            # "I can X" -> "X I can"
            (r"^i can (.+)$", r"\1 I can"),
            # "You must X" -> "X you must"
            (r"^you must (.+)$", r"\1 you must"),
            # "I must X" -> "X I must"
            (r"^i must (.+)$", r"\1 I must"),
            # "There is X" -> "X there is"
            (r"^there is (.+)$", r"\1 there is"),
            # "There are X" -> "X there are"
            (r"^there are (.+)$", r"\1 there are"),
        ]

    def transform_sentence(self, sentence: str) -> str:
        """Transform a single sentence to Yoda-like speech."""
        if not sentence.strip():
            return sentence

        # Clean and normalize
        sentence = sentence.strip().lower()

        # Remove punctuation for processing, but remember it
        punctuation = ""
        if sentence and sentence[-1] in "!?.":
            punctuation = sentence[-1]
            sentence = sentence[:-1]

        # Try pattern matching first
        for pattern, replacement in self.patterns:
            if re.match(pattern, sentence, re.IGNORECASE):
                result = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                return self._capitalize_first(result) + punctuation

        # Fallback: simple word reversal for short sentences
        words = sentence.split()
        if len(words) <= 6:  # Only reverse short sentences
            reversed_words = words[::-1]
            result = " ".join(reversed_words)
            return self._capitalize_first(result) + punctuation

        # For longer sentences, just return as-is with capitalization
        return self._capitalize_first(sentence) + punctuation

    def transform_text(self, text: str) -> str:
        """Transform entire text, handling multiple sentences."""
        if not text.strip():
            return text

        # Split by sentence endings
        sentences = re.split(r"([.!?])", text)

        result_parts = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] if i < len(sentences) else ""
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""

            if sentence.strip():
                transformed = self.transform_sentence(sentence + punctuation)
                result_parts.append(transformed)
            elif punctuation:
                result_parts.append(punctuation)

        return " ".join(result_parts)

    def _capitalize_first(self, text: str) -> str:
        """Capitalize the first letter of the text."""
        if not text:
            return text
        return text[0].upper() + text[1:] if len(text) > 1 else text.upper()
