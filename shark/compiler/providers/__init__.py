from __future__ import annotations

import warnings

from libcst import matchers, CSTNode


class ReturnFinder(matchers.MatcherDecoratableVisitor):
    _returns = []
    func_node = None

    @matchers.visit(matchers.FunctionDef())
    def visit_(self, node):
        self.func_node = node
        return True

    @matchers.visit(matchers.Return())
    def visit_(self, node):
        self._returns.append(node)
        return False

    def __call__(self, node: CSTNode):
        node.visit(self)
        if len(self._returns) == 0:
            warnings.warn(f"no return for {self.func_node.name}")
            return None

        function_return = self._returns.pop()
        if len(self._returns) > 0:
            raise Exception(f"multiple return sites unsupported {self.func_node.name}")
        return function_return
