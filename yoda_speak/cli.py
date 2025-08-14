"""Command line interface for Yoda speech transformation."""

import sys
from typing import NoReturn

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
        """Initialize the CLI with a transformer."""
        self.transformer = YodaTransformer()

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
        return self.transformer.transform_text(text)


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
