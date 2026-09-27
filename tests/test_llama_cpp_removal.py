from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    ROOT / "__init__.py",
    ROOT / "nnodes.py",
    ROOT / "py" / "image_captioning_node.py",
    ROOT / "requirements.txt",
    ROOT / "pyproject.toml",
]


def test_runtime_has_no_llama_cpp_dependency():
    for path in RUNTIME_FILES:
        assert "llama_cpp" not in path.read_text(), path


def test_llava_node_is_no_longer_registered():
    source = (ROOT / "py" / "image_captioning_node.py").read_text()
    assert '"Llava Clip Loader [n-suite]"' not in source
    assert "Llava15ChatHandler" not in source


def test_readme_announces_breaking_change_and_model_downloads():
    readme = (ROOT / "README.md").read_text()
    assert "Breaking change in 1.2.0" in readme
    assert "was never downloaded automatically" in readme
    assert "Downloads happen on first model use" in readme
    assert "3.72 GB" in readme
    assert "0.37 GB" in readme
    assert "git checkout ae7cc84" in readme


def test_dependencies_allow_current_comfyui_versions():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "moviepy>=2.2.1,<3" in requirements
    assert '"moviepy>=2.2.1,<3"' in pyproject
    assert "huggingface-hub" in requirements
    assert "transformers>=4.36.2,<5" in requirements
    assert "timm>=1.0.22" in requirements
    assert "accelerate>=1.0,<2" in requirements
    assert "scikit-build" not in requirements


def test_extension_does_not_install_packages_during_import():
    bootstrap = (ROOT / "__init__.py").read_text()
    assert "check_and_install" not in bootstrap


def test_frontend_uses_managed_dom_widgets():
    widgets = (ROOT / "js" / "extended_widgets.js").read_text()
    assert "addDOMWidget" in widgets
    assert "addCustomWidget" not in widgets
    assert "onDrawBackground" not in widgets
    assert "graph._nodes" not in widgets


def test_dynamic_widgets_use_current_removal_api_and_node_id():
    dynamic_prompt = (ROOT / "js" / "dynamicPrompt.js").read_text()
    gpt_sampler = (ROOT / "js" / "gptSampler.js").read_text()
    assert 'nodeData.name !== "DynamicPrompt [n-suite]"' in dynamic_prompt
    assert "removeWidget(widget)" in dynamic_prompt
    assert "removeWidget(widget)" in gpt_sampler


def test_external_repositories_and_model_archive_are_pinned():
    bootstrap = (ROOT / "__init__.py").read_text()
    assert 'RIFE_REVISION = "a8a8035323b1c1a4a20753c751780e5b0a879455"' in bootstrap
    captioning = (ROOT / "py" / "image_captioning_node.py").read_text()
    assert 'MOONDREAM_REVISION = "f6e9da68e8f1b78b8f3ee10905d56826db7a5802"' in captioning
    assert 'RIFE_MODEL_REVISION = "572480112b87f9bfbff7579b8a38b483766e455f"' in bootstrap
    assert "/raw/main/RIFE_trained_model" not in bootstrap
    assert "repo.git.checkout(revision)" in bootstrap


def test_registry_publish_is_manual_after_nightly_validation():
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow

    readme = (ROOT / "README.md").read_text()
