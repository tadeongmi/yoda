"""Command line interface for Yoda speech transformation."""

import sys
from typing import NoReturn

from .nlp_transformer import NLPYodaTransformer
from .transformer import YodaTransformer


YODA_ASCII = """
    ⠀⠀⠀⠀⠀⠀⠀⢀⣤⣶⣿⣿⣿⣿⣿⣶⣤⡀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀
    ⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀
    ⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀
    ⠀⠀⠀⢸⣿⣿⣿⡿⠛⢿⣿⣿⣿⣿⣿⡿⠛⢿⣿⣿⣿⡇⠀⠀⠀
    ⠀⠀⠀⠀⢿⣿⣿⡇⠀⠈⢿⣿⣿⣿⡿⠁⠀⢸⣿⣿⡿⠀⠀⠀⠀
    ⠀⠀⠀⠀⠘⣿⣿⣿⡄⠀⠀⠻⣿⣿⠟⠀⠀⢠⣿⣿⣿⠃⠀⠀⠀
    ⠀⠀⠀⠀⠀⠘⢿⣿⣿⣦⡀⠀⢈⡁⠀⢀⣴⣿⣿⡿⠃⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⣿⣿⣿⣿⡿⠟⠉⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

    🌟 Welcome, young Padawan. Speak, and Yoda-like your words will become.
    📖 Type 'quit' or 'exit' to leave you must.

"""


class YodaCLI:
    """Command line interface for Yoda speech transformation."""

    def __init__(self) -> None:
        """Initialize the CLI with transformers (NLP primary, regex fallback)."""
        try:
            self.transformer = NLPYodaTransformer()
            self.fallback_transformer = YodaTransformer()
            self.use_nlp = True
            print("🧠 Enhanced NLP mode activated!")
        except RuntimeError as e:
            print(f"⚠️  NLP mode unavailable: {e}")
            print("📝 Falling back to regex-based transformation")
            self.transformer = YodaTransformer()
            self.fallback_transformer = None
            self.use_nlp = False

    def show_greeting(self) -> None:
        """Display the Yoda ASCII art and greeting."""
        print(YODA_ASCII)

    def run_interactive(self) -> None:
        """Run the interactive CLI mode."""
        self.show_greeting()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Yoda: Leave, you do. Strong with the Force you have become.")
                    break

                transformed = self.transformer.transform_text(user_input)
                print(f"Yoda: {transformed}")

            except KeyboardInterrupt:
                print("\nYoda: Interrupted, we were. Farewell, young Padawan.")
                break
            except EOFError:
                print("\nYoda: Strong with the Force you have become. Farewell.")
                break

    def transform_single(self, text: str) -> str:
        """Transform a single text input and return the result."""
        try:
            result = self.transformer.transform_text(text)
            # If NLP gives same result as input and we have fallback, try fallback
            if (
                self.use_nlp
                and self.fallback_transformer
                and result.lower().strip() == text.lower().strip()
            ):
                fallback_result = self.fallback_transformer.transform_text(text)
                if fallback_result.lower().strip() != text.lower().strip():
                    return fallback_result
            return result
        except Exception:
            # If NLP fails, use fallback
            if self.fallback_transformer:
                return self.fallback_transformer.transform_text(text)
            return text


def main() -> NoReturn:
    """Main entry point for the CLI application."""
    cli = YodaCLI()

    if len(sys.argv) > 1:
        # Command line argument mode
        text = " ".join(sys.argv[1:])
        result = cli.transform_single(text)
        print(result)
    else:
        # Interactive mode
        cli.run_interactive()

    sys.exit(0)


if __name__ == "__main__":
    main()
