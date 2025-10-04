import nox  # type: ignore


@nox.session(name="lint", venv_backend="uv")
@nox.parametrize("python", ["3.10"])
def test_lint(session):
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install(
        *pyproject["project"]["dependencies"],
        *pyproject["project"]["optional-dependencies"]["dev"],
        *pyproject["project"]["optional-dependencies"]["onnx-cpu"],
    )
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
    session.run("ty", "check", ".")


@nox.session(name="test-python", venv_backend="uv")
@nox.parametrize("python", ["3.10", "3.11", "3.12", "3.13"])
@nox.parametrize("torch", ["2.8.0"])
def test_python(session, torch):
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install(
        f"torch=={torch}",
        *pyproject["project"]["dependencies"],
        *pyproject["project"]["optional-dependencies"]["dev"],
        *pyproject["project"]["optional-dependencies"]["onnx-cpu"],
    )
    session.run("pytest")


@nox.session(name="test-torch", venv_backend="uv")
@nox.parametrize("python", ["3.11"])
@nox.parametrize("torch", ["2.4", "2.6", "2.8.0"])
def test_torch(session, torch):
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install(
        f"torch=={torch}",
        *pyproject["project"]["dependencies"],
        *pyproject["project"]["optional-dependencies"]["dev"],
        *pyproject["project"]["optional-dependencies"]["onnx-cpu"],
    )
    session.run("pytest")
