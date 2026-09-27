import importlib.util
import os
import sys
import traceback
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

RIFE_REPOSITORY = "https://github.com/hzwer/Practical-RIFE.git"
RIFE_REVISION = "a8a8035323b1c1a4a20753c751780e5b0a879455"
RIFE_MODEL_REVISION = "572480112b87f9bfbff7579b8a38b483766e455f"


def clone_at_revision(repo_class, repository, destination, revision):
    """Clone an external repository once and keep it on a tested revision."""
    repo = repo_class.clone_from(repository, destination) if not os.path.exists(destination) else repo_class(destination)
    if repo.head.commit.hexsha != revision:
        repo.git.checkout(revision)
    return repo

# Pytest imports repository-level __init__.py files while discovering tests. A
# standalone import has no package context and, unlike ComfyUI, cannot resolve
# the extension's relative imports. Leave the mappings empty in that context.
if __package__:
    from .nnodes import color, downloader, get_commit, get_ext_dir, init

    if init():
        print("------------------------------------------")
        print(f"{color.BLUE}### N-Suite Revision:{color.END} {color.GREEN}{get_commit()} {color.END}")
        py = Path(get_ext_dir("py"))
        files = list(py.glob("*.py"))
        print(
            f"{color.YELLOW}N-Suite 1.2 removed the llama.cpp/GGUF and LLaVA nodes. "
            f"Use commit ae7cc84 to keep the legacy nodes.{color.END}"
        )

        from git import Repo

        rife_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "libs", "rifle")
        clone_at_revision(Repo, RIFE_REPOSITORY, rife_path, RIFE_REVISION)

        if not os.path.exists(os.path.join(rife_path, "train_log")):
            downloader(
                f"https://raw.githubusercontent.com/Nuked88/DreamingAI/{RIFE_MODEL_REVISION}/RIFE_trained_model_v4.7.zip"
            )

        # Code based on pysssss's repository.
        for file in files:
            try:
                name = os.path.splitext(file)[0]
                spec = importlib.util.spec_from_file_location(name, os.path.join(py, file))
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
                if mappings is not None:
                    NODE_CLASS_MAPPINGS.update(mappings)
                    display_mappings = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)
                    if display_mappings is not None:
                        NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)
            except Exception:
                traceback.print_exc()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
