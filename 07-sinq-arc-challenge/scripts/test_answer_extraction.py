import importlib.util
from pathlib import Path

script_path = Path(__file__).with_name("run_aime.py")
spec = importlib.util.spec_from_file_location("run_aime", script_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_plain_integer_answer():
    assert module.get_prediction(" 70\n") == "70"


def test_explicit_answer_line():
    assert module.get_prediction("Work...\nANSWER: 70") == "70"


def test_final_boxed_answer():
    assert module.get_prediction(r"Work \boxed{21}; final \boxed{70}") == "70"


def test_final_sum_expression():
    assert module.get_prediction(r"Step: Sum = $21 + 49 = 70$") == "70"


def test_conventional_answer_sentence():
    assert module.get_prediction("Therefore, the answer is 70.") == "70"


def test_sum_sentence():
    assert module.get_prediction("The sum is 21 + 49 = 70.") == "70"


def test_unlabeled_reasoning_integer_is_not_guessed():
    assert module.get_prediction("Tried 21, 49, and 56 but did not conclude.") is None


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
