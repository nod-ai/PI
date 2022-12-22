# /Users/mlevental/dev_projects/SharkPy/venv/lib/python3.10/site-packages/pyccolo/__init__.py

before_if_body = TraceEvent.before_if_body
after_if_iter = TraceEvent.after_if_iter

# /Users/mlevental/dev_projects/SharkPy/venv/lib/python3.10/site-packages/pyccolo/trace_events.py

before_if_body = "before_if_body"
after_if_iter = "after_if_iter"


# /Users/mlevental/dev_projects/SharkPy/venv/lib/python3.10/site-packages/pyccolo/stmt_inserter.py

def _handle_if_body(
        self, node: ast.If, orig_body: List[ast.AST]
) -> List[ast.AST]:
    if_node_copy = cast(
        ast.If, self.orig_to_copy_mapping[id(node)]
    )
    if self.global_guards_enabled:
        if_node_copy = self._global_nonlocal_stripper.visit(if_node_copy)
        if_guard = make_guard_name(if_node_copy)
        self.register_guard(if_guard)
    else:
        if_guard = None
    with fast.location_of(if_node_copy):
        before_if_evt = TraceEvent.before_if_body
        after_if_evt = TraceEvent.after_if_iter
        if self.handler_predicate_by_event[after_if_evt](node):
            ret: List[ast.AST] = [
                fast.Try(
                    body=orig_body,
                    handlers=[],
                    orelse=[],
                    finalbody=[
                        _get_parsed_append_stmt(
                            cast(ast.stmt, if_node_copy),
                            evt=after_if_evt,
                            guard=fast.Str(if_guard),
                        ),
                    ],
                ),
            ]
        else:
            ret = orig_body
        if self.global_guards_enabled:
            ret = [
                fast.If(
                    test=make_composite_condition(
                        [
                            make_test(TRACING_ENABLED),
                            make_test(if_guard),
                            self.emit(
                                before_if_evt, node, ret=fast.NameConstant(True)
                            )
                            if self.handler_predicate_by_event[before_if_evt](
                                node
                            )
                            else None,
                        ]
                    ),
                    body=ret,
                    orelse=if_node_copy.body,
                )
            ]
        return ret
