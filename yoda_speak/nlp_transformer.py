"""Advanced Yoda speech transformation using NLP - Simplified approach."""

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

        # Try specific patterns based on sentence structure
        transformed = self._yoda_transform(doc)

        if transformed != clean_sentence:
            return self._capitalize_first(transformed) + punctuation

        # Fallback to simple word reversal for very short sentences (≤4 words)
        if len(doc) <= 4:
            words = [token.text for token in doc]
            reversed_words = words[::-1]
            result = " ".join(reversed_words)
            return self._capitalize_first(result) + punctuation

        # Return original with proper capitalization if no transformation worked
        return self._capitalize_first(clean_sentence) + punctuation

    def _yoda_transform(self, doc: Doc) -> str:
        """Main Yoda transformation logic."""
        if len(doc) == 0:
            return doc.text

        # Find key sentence components
        subject = self._find_subject(doc)
        root_verb = self._find_root_verb(doc)

        # Pattern 1: "You/I are/am ADJECTIVE" -> "ADJECTIVE you/I are/am"
        if self._is_copula_with_adjective(doc, subject, root_verb):
            return self._transform_copula_adjective(doc, subject, root_verb)

        # Pattern 2: "You/I VERB OBJECT" -> "VERB OBJECT you/I"
        if self._has_direct_object(doc, subject, root_verb):
            return self._transform_with_object(doc, subject, root_verb)

        # Pattern 3: "You/I will/can/must VERB" -> "VERB you/I will/can/must"
        if self._has_modal_auxiliary(doc, subject, root_verb):
            return self._transform_modal_construction(doc, subject, root_verb)

        # No specific pattern matched
        return doc.text

    def _is_copula_with_adjective(
        self, doc: Doc, subject: Optional[Token], root_verb: Optional[Token]
    ) -> bool:
        """Check if this is a 'X is/are ADJECTIVE' pattern."""
        if not root_verb or not subject:
            return False
        if root_verb.lemma_ != "be":
            return False
        # Look for adjectives or predicates after the verb
        for token in doc:
            if (
                token.i > root_verb.i
                and token.pos_ in ["ADJ"]
                or token.dep_ in ["acomp", "attr"]
            ):
                return True
        return False

    def _transform_copula_adjective(
        self, doc: Doc, subject: Optional[Token], root_verb: Optional[Token]
    ) -> str:
        """Transform 'You are strong' -> 'Strong you are'."""
        if not subject or not root_verb:
            return doc.text

        # Collect all tokens
        subject_phrase = []  # Subject and its modifiers
        verb_phrase = []  # The copula and auxiliaries
        predicate = []  # Adjectives, predicates after the verb
        other = []  # Everything else (determiners, etc.)

        for token in doc:
            if token == subject:
                subject_phrase.append(token.text)
            elif token == root_verb or (
                token.dep_ == "aux" and token.head == root_verb
            ):
                verb_phrase.append(token.text)
            elif token.i > root_verb.i and (
                token.pos_ == "ADJ" or token.dep_ in ["acomp", "attr", "pobj"]
            ):
                predicate.append(token.text)
            elif token.dep_ == "det" and any(
                t.head == token for t in doc if t.dep_ in ["acomp", "attr", "pobj"]
            ):
                predicate.append(token.text)  # Determiners that modify predicate nouns
            elif token.i < subject.i:
                other.append(token.text)

        if predicate:
            # Reorder: [other] predicate subject verb
            result_parts = []
            if other:
                result_parts.extend(other)
            result_parts.extend(predicate)
            result_parts.extend(subject_phrase)
            result_parts.extend(verb_phrase)
            return " ".join(result_parts)

        return doc.text

    def _has_direct_object(
        self, doc: Doc, subject: Optional[Token], root_verb: Optional[Token]
    ) -> bool:
        """Check if sentence has a direct object."""
        if not root_verb:
            return False
        for token in doc:
            if token.dep_ == "dobj" and token.head == root_verb:
                return True
        return False

    def _transform_with_object(
        self, doc: Doc, subject: Optional[Token], root_verb: Optional[Token]
    ) -> str:
        """Transform 'I love you' -> 'Love you I' or 'I will help you' -> 'Help you I will'."""
        if not subject or not root_verb:
            return doc.text

        # Find direct object
        direct_object = None
        for token in doc:
            if token.dep_ == "dobj" and token.head == root_verb:
                direct_object = token
                break

        if not direct_object:
            return doc.text

        # Collect components
        subject_tokens = [subject.text]
        verb_tokens = [root_verb.text]
        object_tokens = [direct_object.text]
        aux_tokens = []
        other_tokens = []

        for token in doc:
            if token in [subject, root_verb, direct_object]:
                continue
            elif token.dep_ == "aux" and token.head == root_verb:
                aux_tokens.append(token.text)
            elif token.i < subject.i:  # Before subject
                other_tokens.append(token.text)

        # Reorder: [other] verb object subject aux
        result_parts = []
        if other_tokens:
            result_parts.extend(other_tokens)
        result_parts.extend(verb_tokens)
        result_parts.extend(object_tokens)
        result_parts.extend(subject_tokens)
        if aux_tokens:
            result_parts.extend(aux_tokens)

        return " ".join(result_parts)

    def _has_modal_auxiliary(
        self, doc: Doc, subject: Optional[Token], root_verb: Optional[Token]
    ) -> bool:
        """Check if sentence has modal auxiliary (will, can, must, etc.)."""
        if not root_verb:
            return False
        for token in doc:
            if (
                token.dep_ == "aux"
                and token.head == root_verb
                and token.lemma_ in ["will", "can", "must", "should", "could", "would"]
            ):
                return True
        return False

    def _transform_modal_construction(
        self, doc: Doc, subject: Optional[Token], root_verb: Optional[Token]
    ) -> str:
        """Transform 'You will learn' -> 'Learn you will'."""
        if not subject or not root_verb:
            return doc.text

        # Find modal auxiliary
        modal = None
        for token in doc:
            if (
                token.dep_ == "aux"
                and token.head == root_verb
                and token.lemma_ in ["will", "can", "must", "should", "could", "would"]
            ):
                modal = token
                break

        if not modal:
            return doc.text

        # Simple reorder: verb subject modal [other]
        other_tokens = []
        for token in doc:
            if token not in [subject, root_verb, modal]:
                other_tokens.append(token.text)

        result_parts = [root_verb.text, subject.text, modal.text]
        if other_tokens:
            result_parts.extend(other_tokens)

        return " ".join(result_parts)

    def _find_subject(self, doc: Doc) -> Optional[Token]:
        """Find the main subject of the sentence."""
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                return token
        return None

    def _find_root_verb(self, doc: Doc) -> Optional[Token]:
        """Find the root verb of the sentence."""
        for token in doc:
            if token.dep_ == "ROOT":
                return token
        return None

    def _capitalize_first(self, text: str) -> str:
        """Capitalize the first letter of the text."""
        if not text:
            return text
        return text[0].upper() + text[1:] if len(text) > 1 else text.upper()
