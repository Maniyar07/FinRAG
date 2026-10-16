"""Cooperative cancellation shared by graph steps and worker threads."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event
from time import monotonic

from langchain_core.callbacks import BaseCallbackHandler


class RequestAborted(RuntimeError):
    pass


@dataclass
class RequestControl:
    deadline: float
    cancelled: Event = field(default_factory=Event)

    def check(self) -> None:
        if self.cancelled.is_set() or monotonic() >= self.deadline:
            raise RequestAborted("The request was cancelled or reached its deadline.")


_CONTROL: ContextVar[RequestControl | None] = ContextVar("finrag_request_control", default=None)


@contextmanager
def request_control(control: RequestControl):
    token = _CONTROL.set(control)
    try:
        control.check()
        yield
    finally:
        _CONTROL.reset(token)


def check_request() -> None:
    control = _CONTROL.get()
    if control is not None:
        control.check()


class CancellationCallback(BaseCallbackHandler):
    raise_error = True

    def on_llm_start(self, *args, **kwargs) -> None:
        check_request()

    def on_chat_model_start(self, *args, **kwargs) -> None:
        check_request()
