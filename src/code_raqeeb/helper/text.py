def split_lines(text: str) -> list[str]:
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    return lines[:-1] if lines[-1] == "" else lines
