from __future__ import annotations

import builtins
import dataclasses
from typing import Optional

import libcst as cst
from libcst import matchers as m
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    ParentNodeProvider,
    PositionProvider,
    ExpressionContextProvider,
    Scope,
)
from libcst.metadata.scope_provider import ScopeVisitor, LocalScope


class ForScope(LocalScope):
    pass


class MyScopeVisitor(ScopeVisitor):
    def __init__(self, provider: MyScopeProvider) -> None:
        super().__init__(provider)

    for_scopes = 0
    if_scopes = 0

    def visit_For(self, node: cst.For) -> Optional[bool]:
        self.for_scopes += 1
        if isinstance(node.target, cst.Tuple):
            for elt in node.target.elements:
                self.scope.record_assignment(elt.value, node)
                self.provider.set_metadata(elt, self.scope)
        else:
            self.scope.record_assignment(node.target.value, node)
            self.provider.set_metadata(node.target, self.scope)

        self.provider.set_metadata(node.body, self.scope)

        for_loop_n = f"for_{self.for_scopes}"
        # nested loops
        if isinstance(self.scope.node, cst.For):
            enclosing_scope = self.scope.name
        else:
            enclosing_scope = get_full_name_for_node(self.scope.node)
        with self._new_scope(ForScope, node, f"{enclosing_scope}.{for_loop_n}"):
            node.body.visit(self)

        return False


class MyScopeProvider(cst.BatchableMetadataProvider[Optional[Scope]]):
    METADATA_DEPENDENCIES = (ExpressionContextProvider,)

    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        visitor = MyScopeVisitor(self)
        node.visit(visitor)
        visitor.infer_accesses()


# def livein_liveout(fn_ast):
#     fn_ast = cst.MetadataWrapper(fn_ast).visit(ScopeTransformer())
#     livein_liveout_visitor = LiveInLiveOutVisitor()
#     cst.MetadataWrapper(fn_ast).visit(livein_liveout_visitor)
#     return livein_liveout_visitor.live_ins, livein_liveout_visitor.live_outs


@dataclasses.dataclass
class LiveInsLiveOuts:
    live_ins: set[str]
    live_outs: set[str]
    scope: str

    def __iter__(self):
        yield from [self.live_ins, self.live_outs, self.scope]


class LiveInLiveOutProvider(
    cst.VisitorMetadataProvider[LiveInsLiveOuts], m.MatcherDecoratableVisitor
):
    METADATA_DEPENDENCIES = (
        MyScopeProvider,
        ParentNodeProvider,
        PositionProvider,
        ExpressionContextProvider,
    )

    for_scopes = 0

    builtins = set(builtins.__dict__.keys())

    def visit_For(self, node: cst.For) -> Optional[bool]:
        scope = self.get_metadata(MyScopeProvider, node.body)
        assignments = set(scope._assignments.keys()) - {node.target.value}
        accesses = scope.accesses._accesses.keys() - {node.target.value} - self.builtins
        live_ins = accesses - assignments

        parent_assignments = set(scope.parent._assignments.keys()) - assignments
        parent_accesses = set(scope.parent.accesses._accesses.keys())
        parent_live_ins = parent_accesses - parent_assignments
        live_outs = assignments.intersection(parent_live_ins)

        self.set_metadata(
            node,
            LiveInsLiveOuts(live_ins, live_outs, scope.name),
        )
        return True

    # def visit_If(self, node: If) -> Optional[bool]:
    #     scope = self.get_metadata(ScopeProvider, node)
    #     return True
    #
    # def visit_Else(self, node: Else) -> Optional[bool]:
    #     scope = self.get_metadata(ScopeProvider, node)
    #     return True
