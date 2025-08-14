# < Yoda Speak CLI

Transform text into Yoda-like speech patterns using advanced NLP or simple regex transformations.

## ( Features

- **>à NLP Mode**: Advanced grammar-aware transformations using spaCy
- **=Ý Regex Mode**: Simple pattern-based transformations 
- **= Auto Mode**: Smart fallback from NLP to regex
- **=¬ Interactive Mode**: Real-time conversation with Yoda
- **<¨ ASCII Art**: Beautiful Yoda greeting
- **¡ Fast**: Lightweight and responsive

## =€ Installation

```bash
git clone https://github.com/tadeongmi/yoda.git
cd yoda
uv install
```

## <¯ Usage

### Command Line Mode

```bash
# Auto mode (default) - NLP with regex fallback
uv run python -m yoda_speak.cli "You are strong"
# Output: Strong You are

# Force NLP mode - Advanced grammar understanding  
uv run python -m yoda_speak.cli --mode nlp "I will help you"
# Output: Help you I will

# Force regex mode - Simple and reliable
uv run python -m yoda_speak.cli --mode regex "Thank you very much" 
# Output: Much very you thank
```

### Interactive Mode

```bash
uv run python -m yoda_speak.cli
# Launches interactive session with Yoda ASCII art
```

### Help

```bash
uv run python -m yoda_speak.cli --help
```

## > Transformation Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `auto` | NLP with regex fallback (default) | General use, best results |
| `nlp` | Advanced grammar-aware parsing | Complex sentences, precise grammar |
| `regex` | Simple pattern matching | Simple sentences, guaranteed results |

## =Ê Mode Comparison

```bash
Input: "You are very strong"

auto:  Strong You are          # NLP transformation
nlp:   Strong You are          # Pure NLP 
regex: Strong very are you     # Regex fallback
```

## =à NLP Features

The NLP mode uses spaCy to understand:
- **Subject-Predicate Inversion**: "You are strong" ’ "Strong you are"
- **Object Fronting**: "I will help you" ’ "Help you I will" 
- **Modal Auxiliaries**: "You must learn" ’ "Learn you must"
- **Dependency Parsing**: Preserves all words and relationships

## =' Dependencies

- **Python 3.12+**
- **spaCy**: Advanced NLP processing
- **en_core_web_sm**: English language model

## <¨ Examples

```bash
# Copula constructions
$ yoda "You are wise"
Wise You are

# Action verbs with objects  
$ yoda "I can sense your fear"
Sense fear I can

# Modal constructions
$ yoda "You will become powerful" 
Become You will powerful

# Complex sentences (preserved when no pattern matches)
$ yoda "The complexity of this sentence makes it difficult"
The complexity of this sentence makes it difficult

# Interactive mode
$ yoda
< Welcome, young Padawan...
You: Hello there
Yoda: There Hello
You: quit
Yoda: Leave, you do. Strong with the Force you have become.
```

## >ê Development

```bash
# Run tests
uv run pytest

# Format code  
uv run ruff format .

# Type checking
uv run pyright

# Install development dependencies
uv add --dev pytest ruff pyright
```

## =Ý License

MIT License - Feel free to use this project however you like!

## > Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and formatting
5. Submit a pull request

---

*Strong with the Force, your code will become.* <