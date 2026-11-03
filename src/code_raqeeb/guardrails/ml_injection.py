# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from gorget.input_scanners import PromptInjection


class GorgetInjectionScanner:
    def __init__(self, threshold: float) -> None:
        self._scanner = PromptInjection(threshold=threshold)

    def is_injection(self, text: str) -> bool:
        _, is_valid, _ = self._scanner.scan(text)
        return not bool(is_valid)
