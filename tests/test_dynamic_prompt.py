import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "py" / "dynamic_prompt_node.py"
SPEC = importlib.util.spec_from_file_location("dynamic_prompt_node", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DynamicPrompt = MODULE.DynamicPrompt


def generate(variable, mode="Fixed", count=1, fixed=""):
    return DynamicPrompt().prompt_generator(variable, "NO", mode, count, fixed)["result"][0]


def test_fixed_prompt_is_combined_with_requested_number_of_unique_tags():
    result = generate("red, green, blue", count=2, fixed="portrait")
    parts = result.split(",")
    assert parts[0] == "portrait"
    assert len(parts[1:]) == 2
    assert len(set(parts[1:])) == 2
    assert set(parts[1:]) <= {"red", "green", "blue"}


def test_requested_tag_count_is_capped_to_available_tags():
    assert set(generate("red,blue", count=20).split(",")) == {"red", "blue"}


def test_empty_variable_prompt_returns_empty_result():
    assert generate("", fixed="portrait") == ""
