from typing import Any, Dict, Optional


class PromptlError(Exception):
    code: str
    message: str
    details: Optional[Dict[str, Any]]

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)

        self.code = code
        self.message = message
        self.details = details

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PromptlError)
            and self.code == other.code
            and self.message == other.message
            and str(self.details) == str(other.details)
        )
