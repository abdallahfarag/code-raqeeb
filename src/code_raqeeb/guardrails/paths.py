from pathlib import PurePosixPath

_MAX_PATH_LENGTH = 300


def safe_repo_path(path: str) -> str | None:
    if not path or len(path) > _MAX_PATH_LENGTH or "\\" in path or "\x00" in path:
        return None
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate.as_posix()
