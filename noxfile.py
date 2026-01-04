import nox  # type: ignore
import os
import requests


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


@nox.session(name="test-examples", venv_backend="uv", default=False)
@nox.parametrize("python", ["3.11"])
@nox.parametrize("torch", ["2.8.0"])
def test_examples(session, torch):
    session.install(f"torch=={torch}")
    session.install("transformers>=4.0.0")
    session.install("-e", ".[dev,onnx-gpu]")
    files_to_download = [
        ("https://models.silero.ai/models/en/en_v5.onnx", "~/Downloads/silero_model_stt_v5.onnx"),
        (
            "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx",
            "~/Downloads/resnet50-v2-7.onnx",
        ),
    ]
    print("Downloading example models...")
    for url, path in files_to_download:
        # session.run("wget", url, "-O", os.path.expanduser(path))
        if os.path.isfile(os.path.expanduser(path)):
            print(f"File {path} already exists, skipping download.")
            continue
        print(f"Downloading {url} to {path}...")
        os.path.name
        with open(os.path.expanduser(path), "wb") as f:
            f.write(requests.get(url).content)

    configs = []
    for file in os.listdir(os.path.join("src", "nvbenjo", "conf")):
        file, ext = os.path.splitext(file)
        if ext in [".yml", ".yaml"]:
            configs.append(file)

    for config in configs:
        print(f"Running example {config}:")
        session.run("nvbenjo", "-cn", config)
