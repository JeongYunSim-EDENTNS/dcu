BOLD = "\033[1m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
BLINK = "\033[5m"
INVERT = "\033[7m"
STRIKETHROUGH = "\033[9m"
RESET = "\033[0m"

def bold(text):
    return f"{BOLD}{text}{RESET}"

def italic(text):
    return f"{ITALIC}{text}{RESET}"

def underline(text):
    return f"{UNDERLINE}{text}{RESET}"

def blink(text):
    return f"{BLINK}{text}{RESET}"

def invert(text):
    return f"{INVERT}{text}{RESET}"

def strikethrough(text):
    return f"{STRIKETHROUGH}{text}{RESET}"
