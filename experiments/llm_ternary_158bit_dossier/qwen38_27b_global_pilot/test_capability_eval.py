from __future__ import annotations

from evaluate_capabilities import extract_last_boxed, normalize_math_answer


def main() -> None:
    assert extract_last_boxed(r"work \boxed{1} then \boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    assert extract_last_boxed("no final box") is None
    assert normalize_math_answer(r"$\left( 3, \frac{\pi}{2} \right)$") == r"(3,\frac{\pi}{2})"
    assert normalize_math_answer("1,024") == "1024"
    assert normalize_math_answer("7.0") == "7"
    print("capability evaluator tests passed")


if __name__ == "__main__":
    main()
