"""Advanced Yoda speech transformation using NLP."""

import re
from typing import Optional

import spacy
from spacy.tokens import Doc, Token


class NLPYodaTransformer:
    """Transform text to mimic Yoda's speech patterns using NLP analysis."""

    def __init__(self) -> None:
        """Initialize the transformer with spaCy NLP model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "SpaCy English model not found. Please install it with: "
                "python -m spacy download en_core_web_sm"
            )

    def transform_text(self, text: str) -> str:
        """Transform entire text, handling multiple sentences."""
        if not text.strip():
            return text

        # Split by sentence endings but preserve them
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

    def transform_sentence(self, sentence: str) -> str:
        """Transform a single sentence using NLP-based analysis."""
        if not sentence.strip():
            return sentence

        # Clean and preserve punctuation
        original_sentence = sentence.strip()
        punctuation = ""
        if original_sentence and original_sentence[-1] in "!?.":
            punctuation = original_sentence[-1]
            clean_sentence = original_sentence[:-1]
        else:
            clean_sentence = original_sentence

        # Parse with spaCy
        doc = self.nlp(clean_sentence)

        # Try different transformation strategies
        transformed = self._try_subject_predicate_inversion(doc)
        if transformed != clean_sentence:
            return self._capitalize_first(transformed) + punctuation

        transformed = self._try_object_fronting(doc)
        if transformed != clean_sentence:
            return self._capitalize_first(transformed) + punctuation

        transformed = self._try_verb_phrase_inversion(doc)
        if transformed != clean_sentence:
            return self._capitalize_first(transformed) + punctuation

        # Fallback to simple word reversal for very short sentences
        if len(doc) <= 6:
            words = [token.text for token in doc]
            reversed_words = words[::-1]
            result = " ".join(reversed_words)
            return self._capitalize_first(result) + punctuation

        # Return original with proper capitalization
        return self._capitalize_first(clean_sentence) + punctuation

    def _try_subject_predicate_inversion(self, doc: Doc) -> str:
        """Try to invert subject and predicate: 'You are strong' -> 'Strong you are'."""
        # Find subject and main verb
        subject = self._find_subject(doc)
        main_verb = self._find_main_verb(doc)

        if not subject or not main_verb:
            return doc.text

        # Look for predicate adjectives or objects
        predicate_parts = []
        for token in doc:
            # Skip the subject and verb
            if token == subject or token == main_verb:
                continue

            # Collect predicate adjectives, objects, and complements
            if token.dep_ in ["attr", "acomp", "dobj", "pobj"] or token.pos_ == "ADJ":
                # Include the token and its children (modifiers)
                predicate_parts.extend(self._get_token_with_children(token, doc))

        if predicate_parts:
            # Remove duplicates while preserving order
            predicate_parts = list(dict.fromkeys(predicate_parts))
            predicate_text = " ".join(predicate_parts)

            # Reconstruct: predicate + subject + verb
            subject_text = subject.text
            verb_text = main_verb.text

            return f"{predicate_text} {subject_text} {verb_text}"

        return doc.text

    def _try_object_fronting(self, doc: Doc) -> str:
        """Try to front direct objects: 'I will help you' -> 'Help you I will'."""
        # Find subject, verb, and object
        subject = self._find_subject(doc)
        main_verb = self._find_main_verb(doc)
        direct_object = self._find_direct_object(doc)

        if not all([subject, main_verb, direct_object]):
            return doc.text

        # Look for auxiliary verbs
        aux_verbs = []
        for token in doc:
            if token.dep_ == "aux" and token.head == main_verb:
                aux_verbs.append(token.text)

        # Construct: verb + object + subject + aux
        verb_phrase = main_verb.text
        object_text = direct_object.text
        subject_text = subject.text
        aux_text = " ".join(aux_verbs)

        if aux_text:
            return f"{verb_phrase} {object_text} {subject_text} {aux_text}"
        else:
            return f"{verb_phrase} {object_text} {subject_text}"

    def _try_verb_phrase_inversion(self, doc: Doc) -> str:
        """Try to invert verb phrases: 'You will learn' -> 'Learn you will'."""
        subject = self._find_subject(doc)
        main_verb = self._find_main_verb(doc)

        if not subject or not main_verb:
            return doc.text

        # Find auxiliary verbs
        aux_verbs = []
        for token in doc:
            if token.dep_ == "aux" and token.head == main_verb:
                aux_verbs.append(token.text)

        if aux_verbs:
            # Construct: main_verb + subject + aux_verbs
            subject_text = subject.text
            verb_text = main_verb.text
            aux_text = " ".join(aux_verbs)
            return f"{verb_text} {subject_text} {aux_text}"

        return doc.text

    def _find_subject(self, doc: Doc) -> Optional[Token]:
        """Find the main subject of the sentence."""
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                return token
        return None

    def _find_main_verb(self, doc: Doc) -> Optional[Token]:
        """Find the main verb (root) of the sentence."""
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                return token
        return None

    def _find_direct_object(self, doc: Doc) -> Optional[Token]:
        """Find the direct object of the sentence."""
        for token in doc:
            if token.dep_ == "dobj":
                return token
        return None

    def _get_token_with_children(self, token: Token, doc: Doc) -> list[str]:
        """Get token text along with its modifier children."""
        result = []

        # Add modifiers that come before
        for child in token.children:
            if child.dep_ in ["amod", "det", "advmod"] and child.i < token.i:
                result.append(child.text)

        # Add the token itself
        result.append(token.text)

        # Add modifiers that come after
        for child in token.children:
            if child.dep_ in ["amod", "det", "advmod"] and child.i > token.i:
                result.append(child.text)

        return result

    def _capitalize_first(self, text: str) -> str:
        """Capitalize the first letter of the text."""
        if not text:
            return text
        return text[0].upper() + text[1:] if len(text) > 1 else text.upper()
