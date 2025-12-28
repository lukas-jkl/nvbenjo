import nox  # type: ignore


@nox.session(name="lint", venv_backend="uv")
@nox.parametrize("python", ["3.10"])
def test_lint(session):
    session.install("-e", ".[dev,onnx-cpu]")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
    session.run("ty", "check", ".")


@nox.session(name="test-python", venv_backend="uv")
@nox.parametrize("python", ["3.10", "3.11", "3.12", "3.13"])
@nox.parametrize("torch", ["2.8.0"])
def test_python(session, torch):
    session.install(f"torch=={torch}")
    session.install("-e", ".[dev,onnx-cpu]")
    session.run("pytest")


@nox.session(name="test-torch", venv_backend="uv")
@nox.parametrize("python", ["3.11"])
@nox.parametrize("torch", ["2.4", "2.6", "2.8.0"])
def test_torch(session, torch):
    session.install(f"torch=={torch}")
    session.install("-e", ".[dev,onnx-cpu]")
    session.run("pytest")
