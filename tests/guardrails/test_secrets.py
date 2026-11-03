import pytest

from code_raqeeb.guardrails.secrets import REDACTED, redact_line, redact_text
from tests.guardrails.fakes import AWS_KEY, GITHUB_PAT, GITHUB_TOKEN, GOOGLE_KEY, JWT, SLACK_TOKEN

CREDENTIAL = "hardcoded credential"
CONNECTION = "password in a connection string"


@pytest.mark.parametrize(
    ("secret", "kind"),
    [
        pytest.param(GITHUB_TOKEN, "GitHub token", id="github-classic"),
        pytest.param(GITHUB_PAT, "GitHub token", id="github-fine-grained"),
        pytest.param(AWS_KEY, "AWS access key", id="aws"),
        pytest.param(GOOGLE_KEY, "Google API key", id="google"),
        pytest.param(SLACK_TOKEN, "Slack token", id="slack"),
        pytest.param(JWT, "JWT", id="jwt"),
    ],
)
def test_token_is_replaced_and_reported_with_its_kind(secret: str, kind: str) -> None:
    assert redact_line(f'var value = "{secret}";') == (f'var value = "{REDACTED}";', [kind])


@pytest.mark.parametrize(
    "header",
    [
        pytest.param("-----BEGIN PRIVATE KEY-----", id="plain"),
        pytest.param("-----BEGIN RSA PRIVATE KEY-----", id="rsa"),
        pytest.param("-----BEGIN EC PRIVATE KEY-----", id="ec"),
    ],
)
def test_private_key_header_is_replaced(header: str) -> None:
    assert redact_line(header) == (REDACTED, ["private key"])


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            'var cs = "Server=db;User Id=sa;Password=hunter2;Pooling=true";',
            f'var cs = "Server=db;User Id=sa;Password={REDACTED};Pooling=true";',
            id="after-semicolon",
        ),
        pytest.param(
            'var cs = "Password=hunter2";', f'var cs = "Password={REDACTED}";', id="after-quote"
        ),
        pytest.param(
            "var cs = 'Pwd=hunter2';", f"var cs = 'Pwd={REDACTED}';", id="pwd-after-quote"
        ),
        pytest.param(
            'var cs = "a=b; password = hunter2;";',
            f'var cs = "a=b; password = {REDACTED};";',
            id="spaces-and-lower-case",
        ),
    ],
)
def test_connection_string_password_keeps_everything_but_the_value(
    line: str, expected: str
) -> None:
    assert redact_line(line) == (expected, [CONNECTION])


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("password=abc", id="bare-assignment"),
        pytest.param("// set password=abc in the form", id="prose"),
    ],
)
def test_password_not_after_a_separator_or_quote_is_untouched(line: str) -> None:
    assert redact_line(line) == (line, [])


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            'private const string ApiKey = "abcdefgh12345";',
            f'private const string ApiKey = "{REDACTED}";',
            id="api-key",
        ),
        pytest.param(
            'var api_key = "abcdefgh12345";', f'var api_key = "{REDACTED}";', id="snake-api-key"
        ),
        pytest.param(
            'var clientSecret = "abcdefgh12345";',
            f'var clientSecret = "{REDACTED}";',
            id="name-contains-secret",
        ),
        pytest.param(
            'var authToken = "abcdefgh12345";',
            f'var authToken = "{REDACTED}";',
            id="name-contains-token",
        ),
        pytest.param(
            'var dbPassword = "abcdefgh12345";',
            f'var dbPassword = "{REDACTED}";',
            id="name-contains-password",
        ),
        pytest.param(
            'Connect(password: "abcdefgh12345");',
            f'Connect(password: "{REDACTED}");',
            id="named-argument",
        ),
        pytest.param('var token = "abcdefgh";', f'var token = "{REDACTED}";', id="exactly-8-chars"),
        pytest.param(
            'const string ApiKey = @"abcdefgh12345";',
            f'const string ApiKey = @"{REDACTED}";',
            id="csharp-verbatim",
        ),
        pytest.param(
            'const string ApiKey = $"abcdefgh12345";',
            f'const string ApiKey = $"{REDACTED}";',
            id="csharp-interpolated",
        ),
    ],
)
def test_named_credential_replaces_only_the_value(line: str, expected: str) -> None:
    assert redact_line(line) == (expected, [CREDENTIAL])


@pytest.mark.parametrize(
    "line",
    [
        pytest.param('var token = "abcdefg";', id="7-chars-is-too-short"),
        pytest.param('token = "Bearer"', id="bearer"),
        pytest.param('var password = "";', id="empty-value"),
        pytest.param("var password = GetPassword();", id="method-call"),
        pytest.param("var tokenCount = tokens.Count;", id="not-a-string"),
        pytest.param('var name = "a perfectly normal string";', id="other-name"),
        pytest.param("SELECT password, token FROM users WHERE id = 1", id="plain-sql"),
        pytest.param("", id="empty-line"),
    ],
)
def test_normal_code_is_untouched(line: str) -> None:
    assert redact_line(line) == (line, [])


def test_every_secret_on_a_line_is_replaced_and_reported() -> None:
    text, kinds = redact_line(f"{AWS_KEY} {GOOGLE_KEY}")

    assert text == f"{REDACTED} {REDACTED}"
    assert kinds == ["AWS access key", "Google API key"]


def test_token_in_a_named_credential_is_hidden_and_reports_both_kinds() -> None:
    assert redact_line(f'var token = "{GITHUB_TOKEN}";') == (
        f'var token = "{REDACTED}";',
        ["GitHub token", CREDENTIAL],
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("one line", id="one-line"),
        pytest.param(f"first\n{AWS_KEY}\nlast", id="secret-in-the-middle"),
        pytest.param(f"\n\n{AWS_KEY}\n\n", id="blank-lines-and-trailing-newlines"),
        pytest.param(f"{AWS_KEY}\r\nsecond\r\n", id="crlf"),
    ],
)
def test_redact_text_keeps_the_exact_line_count(text: str) -> None:
    assert redact_text(text).count("\n") == text.count("\n")


def test_redact_text_changes_only_the_lines_with_secrets() -> None:
    text = f'keep\nvar value = "{AWS_KEY}";\nvar other = "ok";\n'

    assert redact_text(text) == f'keep\nvar value = "{REDACTED}";\nvar other = "ok";\n'
