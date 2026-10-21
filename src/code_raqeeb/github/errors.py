from datetime import datetime


class GitHubError(Exception):
    pass


class InvalidPrUrlError(GitHubError, ValueError):
    def __init__(self, url: str) -> None:
        super().__init__(f"Not a GitHub pull request URL: {url!r}")
        self.url = url


class PrNotFoundError(GitHubError):
    pass


class PrivateRepoError(GitHubError):
    pass


class PrTooLargeError(GitHubError):
    pass


class RateLimitError(GitHubError):
    def __init__(self, reset_at: datetime | None) -> None:
        when = reset_at.isoformat() if reset_at else "later"
        super().__init__(f"GitHub rate limit reached; try again after {when}")
        self.reset_at = reset_at


class GitHubApiError(GitHubError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
