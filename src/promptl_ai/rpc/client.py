import os
import random
import sys
from typing import List, Tuple

import wasmtime as wasm

from promptl_ai.rpc.errors import RPCError
from promptl_ai.rpc.payloads import Parameters
from promptl_ai.rpc.types import Call, Error, ErrorCode, Procedure, Result, _Calls, _Results
from promptl_ai.util import Model


class Pipe(Model):
    stdin: str
    stdout: str
    stderr: str


class ClientOptions(Model):
    module_path: str
    working_dir: str


class Client:
    options: ClientOptions

    engine: wasm.Engine
    linker: wasm.Linker
    store: wasm.Store
    module: wasm.Module

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

    # NOTE: this is needed to be thread-safe and avoid race condition on the file
    # system, because we cannot reuse the same file io for multiple calls yet
    def _open_pipe(self) -> Pipe:
        execution_id = str(random.randint(0, sys.maxsize))
        execution_dir = os.path.join(self.options.working_dir, execution_id)
        os.makedirs(execution_dir, exist_ok=True)

        stdin = os.path.join(execution_dir, "stdin")
        stdout = os.path.join(execution_dir, "stdout")
        stderr = os.path.join(execution_dir, "stderr")

        return Pipe(stdin=stdin, stdout=stdout, stderr=stderr)

    def _close_pipe(self, pipe: Pipe):
        if os.path.exists(pipe.stdin):
            os.remove(pipe.stdin)
        if os.path.exists(pipe.stdout):
            os.remove(pipe.stdout)
        if os.path.exists(pipe.stderr):
            os.remove(pipe.stderr)

    # NOTE: this is needed because we cannot reuse the same WASM instance yet
    def _instantiate(self, pipe: Pipe) -> wasm.Instance:
        config = wasm.WasiConfig()
        config.argv = []
        config.env = {}.items()
        config.stdin_file = pipe.stdin
        config.stdout_file = pipe.stdout
        config.stderr_file = pipe.stderr
        self.store.set_wasi(config)

        return self.linker.instantiate(self.store, self.module)

    def _send(self, pipe: Pipe, data: bytes):
        with open(pipe.stdin, "wb") as file:
            file.write(data + b"\n")

    def _receive(self, pipe: Pipe) -> Tuple[bytes, bytes]:
        with open(pipe.stdout, "rb") as file:
            out = file.read().strip()

        with open(pipe.stderr, "rb") as file:
            err = file.read().strip()

        return out, err

    def _execute(self, calls: List[Call]) -> List[Result]:
        pipe = None

        try:
            pipe = self._open_pipe()
            self._send(pipe, _Calls.dump_json(calls))
            instance = self._instantiate(pipe)
            instance.exports(self.store)["_start"](self.store)  # pyright: ignore [reportCallIssue]
            out, err = self._receive(pipe)
            if err:
                raise RPCError(
                    Error(
                        code=ErrorCode.UnknownError,
                        message=err.decode(),
                    )
                )

            return _Results.validate_json(out)

        except Exception as exception:
            if isinstance(exception, RPCError):
                raise exception

            raise RPCError(
                Error(
                    code=ErrorCode.ExecuteError,
                    message=str(exception),
                )
            ) from exception

        finally:
            if pipe:
                self._close_pipe(pipe)

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
