"""Build-time download only. Runtime has no network and never invokes this file."""
import ast
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def pin_default_revision(snapshot):
    # ML's from_pretrained(repo) asks for main. A SHA-only download does not create this alias.
    snapshot = Path(snapshot)
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(snapshot.name, encoding="utf-8")


def main():
    pipeline = Path(__file__).resolve().parents[1] / "solution" / "pipeline.py"
    tree = ast.parse(pipeline.read_text(encoding="utf-8-sig"))
    repo = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DET_REPO" for target in statement.targets):
            repo = ast.literal_eval(statement.value)
    if not isinstance(repo, str):
        raise RuntimeError("Cannot discover DET_REPO; update service weight preparation for the ML interface")
    revision = os.environ.get("DFINE_REVISION") or None
    snapshot = snapshot_download(repo_id=repo, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "README.md", "LICENSE*"])
    pin_default_revision(snapshot)
    print(f"Model cached: {repo} at {Path(snapshot).name}", flush=True)


if __name__ == "__main__":
    main()
