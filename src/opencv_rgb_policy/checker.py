from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


_CALL_ARGUMENTS = 2


class Color(Enum):
    BGR = "BGR"
    RGB = "RGB"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Finding:
    line: int
    column: int
    code: str
    message: str


@dataclass(frozen=True)
class _Value:
    color: Color
    from_opencv: bool = False
    name: str | None = None


def _path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _path(node.value)
        if parent is not None:
            return (*parent, node.attr)
    return None


def _is_call(node: ast.AST, names: set[tuple[str, ...]]) -> bool:
    path = _path(node)
    return path in names


def _constant_name(node: ast.AST) -> str | None:
    path = _path(node)
    if path is None:
        return None
    return path[-1]


def _read_value(call: ast.Call, reads: set[tuple[str, ...]]) -> _Value | None:
    if not _is_call(call.func, reads):
        return None

    flag = call.args[1] if len(call.args) >= _CALL_ARGUMENTS else None
    flag_name = _constant_name(flag) if flag is not None else None
    if flag_name == "IMREAD_COLOR_RGB":
        color = Color.RGB
    elif flag_name in {None, "IMREAD_COLOR", "IMREAD_COLOR_BGR"}:
        color = Color.BGR
    else:
        color = Color.UNKNOWN
    return _Value(color=color, from_opencv=True)


def _conversion_name(call: ast.Call, conversions: set[tuple[str, ...]]) -> str | None:
    if not _is_call(call.func, conversions):
        return None
    if len(call.args) < _CALL_ARGUMENTS:
        return None
    return _constant_name(call.args[1])


class _ScopeChecker:
    def __init__(self, tree: ast.AST) -> None:
        self.findings: list[Finding] = []
        self._seen: set[tuple[int, int, str]] = set()
        self._scope_stack: list[ast.AST] = []
        self._read_names: set[tuple[str, ...]] = {
            ("cv2", "imread"),
            ("cv2", "imdecode"),
        }
        self._conversion_names: set[tuple[str, ...]] = {("cv2", "cvtColor")}
        self._encode_names: set[tuple[str, ...]] = {("cv2", "imencode")}
        self._visit_scope(tree, {})

    def _add(self, node: ast.AST, code: str, message: str) -> None:
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0)
        key = (line, column, code)
        if key not in self._seen:
            self._seen.add(key)
            self.findings.append(Finding(line, column, code, message))

    def _register_imports(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "cv2":
                    local = alias.asname or "cv2"
                    self._read_names.update({(local, "imread"), (local, "imdecode")})
                    self._conversion_names.add((local, "cvtColor"))
                    self._encode_names.add((local, "imencode"))
        elif isinstance(statement, ast.ImportFrom) and statement.module == "cv2":
            for alias in statement.names:
                local = alias.asname or alias.name
                if alias.name in {"imread", "imdecode"}:
                    self._read_names.add((local,))
                elif alias.name == "cvtColor":
                    self._conversion_names.add((local,))
                elif alias.name == "imencode":
                    self._encode_names.add((local,))

    def _visit_scope(self, node: ast.AST, initial: dict[str, _Value]) -> None:
        body = getattr(node, "body", None)
        if isinstance(body, list):
            self._scope_stack.append(node)
            self._visit_block(body, dict(initial))
            self._scope_stack.pop()

    def _visit_block(self, statements: list[ast.stmt], state: dict[str, _Value]) -> None:
        for statement in statements:
            self._visit_statement(statement, state)

    def _visit_statement(self, statement: ast.stmt, state: dict[str, _Value]) -> None:
        self._register_imports(statement)
        if self._visit_nested_scope(statement):
            return
        if self._visit_assignment(statement, state):
            return
        if self._visit_branch(statement, state):
            return
        if self._visit_context(statement, state):
            return
        self._inspect_calls(statement, state)

    def _visit_nested_scope(self, statement: ast.stmt) -> bool:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_scope(statement, {})
            return True
        if isinstance(statement, ast.ClassDef):
            for item in statement.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._visit_scope(item, {})
            return True
        return False

    def _visit_assignment(self, statement: ast.stmt, state: dict[str, _Value]) -> bool:
        if isinstance(statement, ast.Assign):
            value = self._value(statement.value, state)
            self._inspect_calls(statement.value, state, _target_names(statement.targets))
            for target in statement.targets:
                self._assign(target, value, state)
            return True
        if isinstance(statement, ast.AnnAssign):
            value = self._value(statement.value, state) if statement.value is not None else _Value(Color.UNKNOWN)
            if statement.value is not None:
                self._inspect_calls(statement.value, state, _target_names([statement.target]))
            self._assign(statement.target, value, state)
            return True
        if isinstance(statement, ast.AugAssign):
            self._inspect_calls(statement.value, state)
            self._assign(statement.target, _Value(Color.UNKNOWN), state)
            return True
        return False

    def _visit_branch(self, statement: ast.stmt, state: dict[str, _Value]) -> bool:
        if isinstance(statement, ast.If):
            self._inspect_calls(statement.test, state)
            body_state = dict(state)
            else_state = dict(state)
            self._visit_block(statement.body, body_state)
            self._visit_block(statement.orelse, else_state)
            self._merge_states(state, body_state, else_state)
            return True
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            return self._visit_loop(statement, state)
        if isinstance(statement, ast.Try):
            self._visit_try(statement, state)
            return True
        return False

    def _visit_loop(self, statement: ast.stmt, state: dict[str, _Value]) -> bool:
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            self._inspect_calls(statement.iter, state)
            self._assign(statement.target, _Value(Color.UNKNOWN), state)
        else:
            self._inspect_calls(statement.test, state)
        loop_state = dict(state)
        self._visit_block(statement.body, loop_state)
        self._visit_block(statement.orelse, state)
        self._merge_states(state, state, loop_state)
        return True

    def _visit_try(self, statement: ast.Try, state: dict[str, _Value]) -> None:
        self._visit_block(statement.body, state)
        for handler in statement.handlers:
            self._visit_block(handler.body, dict(state))
        self._visit_block(statement.orelse, state)
        self._visit_block(statement.finalbody, state)

    def _visit_context(self, statement: ast.stmt, state: dict[str, _Value]) -> bool:
        if not isinstance(statement, (ast.With, ast.AsyncWith)):
            return False
        for item in statement.items:
            self._inspect_calls(item.context_expr, state)
            if item.optional_vars is not None:
                self._assign(item.optional_vars, _Value(Color.UNKNOWN), state)
        self._visit_block(statement.body, state)
        return True

    @staticmethod
    def _merge_states(target: dict[str, _Value], left: dict[str, _Value], right: dict[str, _Value]) -> None:
        for name in set(left) | set(right):
            left_value = left.get(name, _Value(Color.UNKNOWN))
            right_value = right.get(name, _Value(Color.UNKNOWN))
            target[name] = left_value if left_value == right_value else _Value(Color.UNKNOWN)

    @staticmethod
    def _assign(target: ast.AST, value: _Value, state: dict[str, _Value]) -> None:
        if isinstance(target, ast.Name):
            state[target.id] = _Value(value.color, value.from_opencv, target.id if value.from_opencv else None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                if isinstance(element, ast.Name):
                    state[element.id] = _Value(Color.UNKNOWN)

    def _value(self, node: ast.AST | None, state: dict[str, _Value]) -> _Value:
        if node is None:
            return _Value(Color.UNKNOWN)
        if isinstance(node, ast.Name):
            return state.get(node.id, _Value(Color.UNKNOWN))
        if not isinstance(node, ast.Call):
            return _Value(Color.UNKNOWN)
        return self._call_value(node, state)

    def _call_value(self, node: ast.Call, state: dict[str, _Value]) -> _Value:
        read = _read_value(node, self._read_names)
        if read is not None:
            return read

        conversion = _conversion_name(node, self._conversion_names)
        if conversion is None or not node.args:
            return _Value(Color.UNKNOWN)
        source = self._value(node.args[0], state)
        if conversion == "COLOR_BGR2RGB":
            if source.color is Color.BGR:
                return _Value(Color.RGB, source.from_opencv, source.name)
        if conversion == "COLOR_RGB2BGR" and source.color is Color.RGB:
            return _Value(Color.BGR, source.from_opencv, source.name)
        return _Value(Color.UNKNOWN)

    def _inspect_calls(self, node: ast.AST, state: dict[str, _Value], target_names: set[str] | None = None) -> None:
        target_names = target_names or set()
        for call in _calls(node):
            conversion = _conversion_name(call, self._conversion_names)
            if conversion == "COLOR_BGR2RGB" and call.args:
                source = self._value(call.args[0], state)
                if (
                    source.color is Color.BGR
                    and source.from_opencv
                    and not self._has_other_use(call, source.name, target_names)
                ):
                    self._add(
                        call,
                        "OPCV001",
                        "OpenCV decoded BGR is converted to RGB without another BGR consumer; use IMREAD_COLOR_RGB.",
                    )

            if _is_call(call.func, self._encode_names):
                image = call.args[1] if len(call.args) >= _CALL_ARGUMENTS else _keyword(call, "img")
                value = self._value(image, state)
                if value.color is Color.RGB:
                    self._add(
                        call,
                        "OPCV002",
                        "cv2.imencode expects BGR input; convert RGB with COLOR_RGB2BGR or keep the BGR image.",
                    )

    def _has_other_use(self, conversion: ast.Call, name: str | None, target_names: set[str]) -> bool:
        if name is None:
            return False
        scope = self._scope_stack[-1]
        for candidate, parents in _name_loads(scope):
            if candidate.id != name:
                continue
            if _is_inside(candidate, conversion):
                continue
            if _is_none_guard(candidate, parents):
                continue
            if name in target_names and _position(candidate) >= _position(conversion):
                continue
            return True
        return False


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _target_names(targets: list[ast.AST]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                names.add(node.id)
    return names


def _position(node: ast.AST) -> tuple[int, int]:
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def _is_inside(node: ast.AST, parent: ast.AST) -> bool:
    if node is parent:
        return True
    for child in ast.iter_child_nodes(parent):
        if _is_inside(node, child):
            return True
    return False


def _calls(node: ast.AST) -> Iterable[ast.Call]:
    return (candidate for candidate in ast.walk(node) if isinstance(candidate, ast.Call))


def _name_loads(node: ast.AST) -> Iterable[tuple[ast.Name, tuple[ast.AST, ...]]]:
    def walk(current: ast.AST, parents: tuple[ast.AST, ...]) -> Iterable[tuple[ast.Name, tuple[ast.AST, ...]]]:
        if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
            yield current, parents
        for child in ast.iter_child_nodes(current):
            yield from walk(child, (*parents, current))

    return walk(node, ())


def _is_none_guard(node: ast.Name, parents: tuple[ast.AST, ...]) -> bool:
    for parent in reversed(parents):
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
            return True
        if isinstance(parent, ast.Compare):
            has_none = any(
                isinstance(value, ast.Constant) and value.value is None for value in (parent.left, *parent.comparators)
            )
            if has_none and all(
                isinstance(operator, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)) for operator in parent.ops
            ):
                return True
        if isinstance(parent, (ast.If, ast.While, ast.BoolOp, ast.Expr)):
            continue
        break
    return False


def check_source(source: str) -> list[Finding]:
    tree = ast.parse(source)
    checker = _ScopeChecker(tree)
    return sorted(checker.findings, key=lambda finding: (finding.line, finding.column, finding.code))
