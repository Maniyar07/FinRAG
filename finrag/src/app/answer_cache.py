"""Bounded reuse of validated answers within one immutable index service."""

from collections import OrderedDict
from concurrent.futures import Future, wait
from copy import deepcopy
from threading import Lock
from time import monotonic
from collections.abc import Callable

from src.generation.answer_guardrails import validate_answer_payload
from src.request_runtime import RequestAborted, check_request
from src.schemas import ChatResult, Decision


class AnswerCache:
    def __init__(self, max_entries: int, ttl_seconds: float):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, ChatResult]] = OrderedDict()
        self._pending: dict[str, Future[ChatResult]] = {}
        self._lock = Lock()

    def get_or_compute(self, key: str, compute: Callable[[], ChatResult]) -> tuple[ChatResult, bool]:
        while True:
            check_request()
            with self._lock:
                entry = self._entries.pop(key, None)
                if entry is not None and entry[0] > monotonic():
                    self._entries[key] = entry
                    return deepcopy(entry[1]), True
                pending = self._pending.get(key)
                if pending is None:
                    pending = self._pending[key] = Future()
                    break
            while not pending.done():
                check_request()
                wait([pending], timeout=0.1)
            try:
                return deepcopy(pending.result()), True
            except RequestAborted:
                # Another request's cancellation must not cancel this caller.
                check_request()
        try:
            result = compute()
            check_request()
            snapshot = deepcopy(result)
            with self._lock:
                if (result.decision == Decision.ANSWERED
                        and result.trace.get("answer_validation", {}).get("valid")
                        and validate_answer_payload(result.answer, result.sources).valid):
                    self._entries[key] = (monotonic() + self.ttl_seconds, snapshot)
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
                self._pending.pop(key)
                pending.set_result(snapshot)
            return result, False
        except BaseException as error:
            with self._lock:
                self._pending.pop(key, None)
                pending.set_exception(error)
            raise
