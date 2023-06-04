import difflib
import re
from textwrap import dedent


def check_correct(correct, op):
    correct = dedent(
        re.sub(r"([%#@]\w+)|(\^bb\d+)|(0x\w+)|dense<.*?>", "%DONT_CARE", correct.strip())
    )
    op = dedent(
        re.sub(r"([%#@]\w+)|(\^bb\d+)|(0x\w+)|dense<.*?>", "%DONT_CARE", str(op))
    )
    diff = list(
        difflib.unified_diff(
            correct.splitlines(),
            op.splitlines(),
            lineterm="",
        )
    )
    assert len(diff) == 0, "\n".join(diff)
