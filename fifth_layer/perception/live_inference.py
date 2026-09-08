"""One in-flight inference; UI polling never waits for model completion."""
from queue import Queue, Empty, Full
from threading import Thread


class LiveInferenceWorker:
    def __init__(self, infer):
        self.infer = infer
        self.requests = Queue(maxsize=1)
        self.results = Queue(maxsize=1)
        self.busy = False
        self.closed = False
        Thread(target=self._run, daemon=True).start()

    def submit(self, frame, timestamp):
        if self.closed or self.busy:
            return False
        self.busy = True
        self.requests.put_nowait((frame.copy(), timestamp))
        return True

    def poll(self):
        try:
            result = self.results.get_nowait()
        except Empty:
            return None
        self.busy = False
        return result

    def close(self):
        self.closed = True
        try:
            self.requests.put_nowait(None)
        except Full:
            pass
        # Never join an in-flight GPU call from the UI thread.

    def _run(self):
        while not self.closed:
            request = self.requests.get()
            if request is None:
                return
            frame, timestamp = request
            try:
                value, error = self.infer(frame), None
            except Exception as exc:
                value, error = None, exc
            if self.closed:
                return
            self.results.put_nowait((frame, timestamp, value, error))
