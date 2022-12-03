src = """\
def f():
    for x in []:
        for y in []:
            x, y
class X:
    def f():
        for x in []:
            for y in []:
                x, y
"""

from textwrap import dedent

from test_scope_provider import get_scope_metadata_provider

m, scopes, positions = get_scope_metadata_provider(src)
src_lines = src.split("\n")

seen_scopes = set()
for scope in scopes.values():
    scope_name = getattr(scope, 'name', 'global')
    if scope_name not in seen_scopes:
        seen_scopes.add(scope_name)
        print(f"\n*** scope {scope_name} ***")
        # An Access records an access of an assignment (expression context LOAD).
        #
        for acc in scope.accesses:
            for ref in acc.referents:
                acc_pos = positions[acc.node].start.line - 1
                ref_pos = positions[ref.node].start.line - 1
                print(dedent(f"""\
                access {acc.node.value} on line {acc_pos + 1}:
                {src_lines[acc_pos]}
                refers to {ref.name} on line {ref_pos + 1}
                {src_lines[ref_pos]}
                """))
