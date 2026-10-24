"""An advisory lock held for the lifetime of a writer, including its children."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def writer_lock(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer owns {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        yield stream.fileno()


def locked_output(argument: str):
    """Protect a function whose output path is supplied by keyword."""
    from functools import wraps

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            path = Path(kwargs[argument])
            with writer_lock(path.with_name(path.name + ".lock")):
                return function(*args, **kwargs)

        return wrapped

    return decorate
