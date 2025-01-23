from typing import Any, Dict, List, Optional

from promptl_ai.util import Adapter, Model, StrEnum


class Procedure(StrEnum):
    ScanPrompt = "scanPrompt"
    RenderPrompt = "renderPrompt"
    CreateChain = "createChain"
    StepChain = "stepChain"


class Call(Model):
    procedure: Procedure
    parameters: Dict[str, Any]


_Calls = Adapter(List[Call])


class ErrorCode(StrEnum):
    ReceiveError = "RECEIVE_ERROR"
    ExecuteError = "EXECUTE_ERROR"
    SendError = "SEND_ERROR"
    ProcedureError = "PROCEDURE_ERROR"
    UnknownError = "UNKNOWN_ERROR"


class Error(Model):
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None


class Result(Model):
    value: Optional[Any] = None
    error: Optional[Error] = None


_Results = Adapter(List[Result])
