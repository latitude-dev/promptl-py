import os
from typing import List, Tuple

import wasmtime as wasm

from promptl_ai.rpc.errors import RPCError
from promptl_ai.rpc.payloads import Parameters
from promptl_ai.rpc.types import Call, Error, ErrorCode, Procedure, Result, _Calls, _Results
from promptl_ai.util import Model


class ClientOptions(Model):
    module_path: str
    working_dir: str


class Client:
    options: ClientOptions

    engine: wasm.Engine
    linker: wasm.Linker
    store: wasm.Store
    module: wasm.Module

    stdin: str
    stdout: str
    stderr: str

    def __init__(self, options: ClientOptions):
        self.options = options

        config = wasm.Config()
        self.engine = wasm.Engine(config)
        self.linker = wasm.Linker(self.engine)
        self.linker.define_wasi()
        self.store = wasm.Store(self.engine)

        with open(self.options.module_path, "rb") as file:
            raw_module = file.read()
        wasm.Module.validate(self.engine, raw_module)
        self.module = wasm.Module(self.engine, raw_module)

        self.stdin = os.path.join(self.options.working_dir, "stdin")
        self.stdout = os.path.join(self.options.working_dir, "stdout")
        self.stderr = os.path.join(self.options.working_dir, "stderr")

        os.makedirs(self.options.working_dir, exist_ok=True)
        with open(self.stdin, "wb") as file:
            file.write(b"")
        with open(self.stdout, "wb") as file:
            file.write(b"")
        with open(self.stderr, "wb") as file:
            file.write(b"")

    def _instantiate(self) -> wasm.Instance:
        config = wasm.WasiConfig()
        config.argv = []
        config.env = {}.items()
        config.stdin_file = self.stdin
        config.stdout_file = self.stdout
        config.stderr_file = self.stderr
        self.store.set_wasi(config)

        return self.linker.instantiate(self.store, self.module)

    def _send(self, data: bytes):
        with open(self.stdin, "wb") as file:
            file.write(data + b"\n")

    def _receive(self) -> Tuple[bytes, bytes]:
        with open(self.stdout, "rb") as file:
            out = file.read().strip()

        with open(self.stderr, "rb") as file:
            err = file.read().strip()

        return out, err

    def _execute(self, calls: List[Call]) -> List[Result]:
        try:
            self._send(_Calls.dump_json(calls))
        except Exception as exception:
            raise RPCError(
                Error(
                    code=ErrorCode.ReceiveError,
                    message=str(exception),
                )
            ) from exception

        try:
            instance = self._instantiate()
            instance.exports(self.store)["_start"](self.store)  # pyright: ignore [reportCallIssue]
        except Exception as exception:
            raise RPCError(
                Error(
                    code=ErrorCode.ExecuteError,
                    message=str(exception),
                )
            ) from exception

        out, err = self._receive()
        if err:
            raise RPCError(
                Error(
                    code=ErrorCode.UnknownError,
                    message=err.decode(),
                )
            )

        if not out:
            raise RPCError(
                Error(
                    code=ErrorCode.SendError,
                    message="No output",
                )
            )

        try:
            results = _Results.validate_json(out)
        except Exception as exception:
            raise RPCError(
                Error(
                    code=ErrorCode.SendError,
                    message=str(exception),
                )
            ) from exception

        return results

    def execute(self, procedure: Procedure, parameters: Parameters) -> Result:
        results = self._execute([Call(procedure=procedure, parameters=parameters.model_dump())])
        if len(results) != 1:
            raise RPCError(
                Error(
                    code=ErrorCode.ExecuteError,
                    message=f"Expected 1 result, got {len(results)}",
                )
            )

        if results[0].error and results[0].error.code != ErrorCode.ProcedureError:
            raise RPCError(results[0].error)

        return results[0]

    def execute_batch(self, procedures: List[Tuple[Procedure, Parameters]]) -> List[Result]:
        calls = [Call(procedure=procedure, parameters=parameters.model_dump()) for procedure, parameters in procedures]

        results = self._execute(calls)
        if len(results) != len(procedures):
            raise RPCError(
                Error(
                    code=ErrorCode.ExecuteError,
                    message=f"Expected {len(procedures)} results, got {len(results)}",
                )
            )

        return results
