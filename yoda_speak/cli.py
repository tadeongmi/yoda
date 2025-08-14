"""Command line interface for Yoda speech transformation."""

import argparse
import sys
from typing import NoReturn

from .nlp_transformer import NLPYodaTransformer
from .transformer import YodaTransformer


YODA_ASCII = """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⠤⠐⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡌⡦⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⣼⡊⢀⠔⠀⠀⣄⠤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣄⣀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣶⠃⠉⠡⡠⠤⠊⠀⠠⣀⣀⡠⠔⠒⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⢟⠿⠛⠛⠁
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡇⠀⠀⠀⠀⠑⠶⠖⠊⠁⠀⠀⠀⡀⠀⠀⠀⢀⣠⣤⣤⡀⠀⠀⠀⠀⠀⢀⣠⣤⣶⣿⣿⠟⡱⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣾⣿⡇⠀⢀⡠⠀⠀⠀⠈⠑⢦⣄⣀⣀⣽⣦⣤⣾⣿⠿⠿⠿⣿⡆⠀⠀⢀⠺⣿⣿⣿⣿⡿⠁⡰⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⣿⣧⣠⠊⣠⣶⣾⣿⣿⣶⣶⣿⣿⠿⠛⢿⣿⣫⢕⡠⢥⣈⠀⠙⠀⠰⣷⣿⣿⣿⡿⠋⢀⠜⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⢿⣿⣿⣿⣿⣰⣿⣿⠿⣛⡛⢛⣿⣿⣟⢅⠀⠀⢿⣿⠕⢺⣿⡇⠩⠓⠂⢀⠛⠛⠋⢁⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀
⠘⢶⡶⢶⣶⣦⣤⣤⣤⣤⣤⣀⣀⣀⣀⡀⠀⠘⣿⣿⣿⠟⠁⡡⣒⣬⢭⢠⠝⢿⡡⠂⠀⠈⠻⣯⣖⣒⣺⡭⠂⢀⠈⣶⣶⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠙⠳⣌⡛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣻⣵⣨⣿⣿⡏⢀⠪⠎⠙⠿⣋⠴⡃⢸⣷⣤⣶⡾⠋⠈⠻⣶⣶⣶⣷⣶⣷⣿⣟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠈⠛⢦⣌⡙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠩⠭⡭⠴⠊⢀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣿⡇⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⠙⠓⠦⣄⡉⠛⠛⠻⢿⣿⣿⣿⣷⡀⠀⠀⠀⠀⢀⣰⠋⠀⠀⠀⠀⠀⣀⣰⠤⣳⣿⣿⣿⣿⣟⠑⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠒⠒⠶⢺⣿⣿⣿⣿⣦⣄⣀⣴⣿⣯⣤⣔⠒⠚⣒⣉⣉⣴⣾⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠹⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣭⣉⣉⣤⣿⣿⣿⣿⣿⣿⡿⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⡁⡆⠙⢶⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣴⣶⣾⣿⣟⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢛⣩⣴⣿⠇⡇⠸⡆⠙⢷⣄⠻⣿⣦⡄⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣎⢻⣿⣿⣿⣿⣿⣿⣿⣭⣭⣭⣵⣶⣾⣿⣿⣿⠟⢰⢣⠀⠈⠀⠀⠙⢷⡎⠙⣿⣦⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⡟⣿⡆⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠟⠛⠋⠁⢀⠇⢸⡇⠀⠀⠀⠀⠈⠁⠀⢸⣿⡆⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡜⡿⡘⣿⣿⣿⣿⣿⣶⣶⣤⣤⣤⣤⣤⣤⣤⣴⡎⠖⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⡄⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠘⢿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠛⠋⡟⠀⠀⣸⣷⣀⣤⣀⣀⣀⣤⣤⣾⣿⣿⣿⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣭⣓⡲⠬⢭⣙⡛⠿⣿⣿⣶⣦⣀⠀⡜⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣭⣛⣓⠶⠦⠥⣀⠙⠋⠉⠉⠻⣄⣀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣆⠐⣦⣠⣷⠊⠁⠀⠀⡭⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢉⣛⡛⢻⡗⠂⠀⢀⣷⣄⠈⢆⠉⠙⠻⢿⣿⣿⣿⣿⣿⠇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⡟⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⣉⢁⣴⣿⣿⣿⣾⡇⢀⣀⣼⡿⣿⣷⡌⢻⣦⡀⠀⠈⠙⠛⠿⠏⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⡄⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠛⠛⢯⡉⠉⠉⠉⠉⠛⢼⣿⠿⠿⠦⡙⣿⡆⢹⣷⣤⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠿⠄⠈⠻⠿⠿⠿⠿⠿⠿⠛⠛⠿⠛⠉⠁⠀⠀⠀⠀⠀⠀⠻⠿⠿⠿⠿⠟⠉⠀⠀⠤⠴⠶⠌⠿⠘⠿⠿⠿⠿⠶⠤⠀⠀⠀⠀

    🌟 Welcome, young Padawan. Speak, and Yoda-like your words will become.
    📖 Type 'quit' or 'exit' to leave you must.

"""


class YodaCLI:
    """Command line interface for Yoda speech transformation."""

    def __init__(self, mode: str = "auto") -> None:
        """Initialize the CLI with specified transformer mode.

        Args:
            mode: Transformer mode - 'nlp', 'regex', or 'auto'
        """
        self.mode = mode
        self.use_nlp = False
        self.fallback_transformer = None

        if mode == "regex":
            # Force regex mode
            self.transformer = YodaTransformer()
            print("📝 Regex mode activated!")
        elif mode == "nlp":
            # Force NLP mode (fail if unavailable)
            try:
                self.transformer = NLPYodaTransformer()
                self.use_nlp = True
                print("🧠 NLP mode activated!")
            except RuntimeError as e:
                print(f"❌ NLP mode unavailable: {e}")
                print("💡 Try using --mode regex instead")
                sys.exit(1)
        else:
            # Auto mode (NLP with regex fallback)
            try:
                self.transformer = NLPYodaTransformer()
                self.fallback_transformer = YodaTransformer()
                self.use_nlp = True
                print("🧠 Enhanced NLP mode activated! (with regex fallback)")
            except RuntimeError as e:
                print(f"⚠️  NLP mode unavailable: {e}")
                print("📝 Falling back to regex-based transformation")
                self.transformer = YodaTransformer()
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
    parser = argparse.ArgumentParser(
        description="Transform text into Yoda-like speech patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yoda "You are strong"                    # Auto mode (NLP with fallback)
  yoda --mode nlp "I will help you"        # Force NLP mode
  yoda --mode regex "Thank you"            # Force regex mode
  yoda                                     # Interactive mode

Modes:
  auto   - Use NLP transformer with regex fallback (default)
  nlp    - Use only NLP transformer (fails if unavailable)  
  regex  - Use only regex transformer (simple, always works)
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "nlp", "regex"],
        default="auto",
        help="Transformation mode (default: auto)",
    )

    parser.add_argument(
        "text",
        nargs="*",
        help="Text to transform (if not provided, enters interactive mode)",
    )

    args = parser.parse_args()

    # Initialize CLI with selected mode
    cli = YodaCLI(mode=args.mode)

    if args.text:
        # Command line argument mode
        text = " ".join(args.text)
        result = cli.transform_single(text)
        print(result)
    else:
        # Interactive mode
        cli.run_interactive()

    sys.exit(0)


if __name__ == "__main__":
    main()
