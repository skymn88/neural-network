"""Bounded, single-flight result cache; one background worker avoids CPU floods."""

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock


class PlotCache:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.pending = {}
        self.lock = Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plot-preload")
        self.preloading = set()

    def get(self, key, compute):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            owner = key not in self.pending
            future = self.pending.setdefault(key, Future())
        if not owner:
            return future.result(timeout=120)
        try:
            value = compute()
            with self.lock:
                self.cache[key] = value
                while len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
            future.set_result(value)
            return value
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self.lock:
                self.pending.pop(key, None)

    def preload(self, key, compute):
        with self.lock:
            if key in self.cache or key in self.pending or key in self.preloading:
                return
            self.preloading.add(key)

        def work():
            try:
                self.get(key, compute)
            except Exception:
                pass  # Foreground retries surface the precise error to the UI.
            finally:
                with self.lock:
                    self.preloading.discard(key)

        self.pool.submit(work)
