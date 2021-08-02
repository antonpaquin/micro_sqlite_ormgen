#! /usr/bin/env python

import ast
import enum
import json
import sys
import textwrap
from typing import *


class Schema:
    models: List["Model"]

    def __init__(self, models: List["Model"]):
        self.models = models

    def to_dict(self):
        return {
            "models": [model.to_dict() for model in self.models],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(models=[Model.from_dict(model) for model in d["models"]])


class Model:
    name: str
    fields: List["Field"]

    def __init__(self, name: str, fields: List["Field"]):
        self.name = name
        self.fields = fields

    def to_dict(self):
        return {"name": self.name, "fields": [field.to_dict() for field in self.fields]}

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls(
            name=d["name"],
            fields=[Field.from_dict(field) for field in d["fields"]],
        )


class Field:
    name: str
    sqltype: "SqlType"
    pkey: bool
    pytype: Optional[str]
    nullable: bool

    def __init__(
        self,
        name: str,
        sqltype: "SqlType",
        pkey: bool = False,
        pytype: Optional[str] = None,
        nullable: bool = False,
    ):
        self.name = name
        self.sqltype = sqltype
        self.pkey = pkey
        self.pytype = pytype
        self.nullable = nullable

    def to_dict(self):
        return {
            "name": self.name,
            "sqltype": self.sqltype.to_s(),
            "pkey": self.pkey,
            "pytype": self.pytype,
            "nullable": self.nullable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Field":
        return cls(
            name=d["name"],
            sqltype=SqlType.from_s(d["sqltype"]),
            pkey=d["pkey"],
            pytype=d["pytype"],
            nullable=d["nullable"],
        )


class SqlType(enum.Enum):
    INTEGER = enum.auto()
    REAL = enum.auto()
    TEXT = enum.auto()
    JSON = enum.auto()

    def to_s(self) -> str:
        return {
            SqlType.INTEGER: "INTEGER",
            SqlType.REAL: "REAL",
            SqlType.TEXT: "TEXT",
            SqlType.JSON: "JSON",
        }[self]

    @staticmethod
    def from_s(s: str) -> "SqlType":
        return {
            "INTEGER": SqlType.INTEGER,
            "REAL": SqlType.REAL,
            "TEXT": SqlType.TEXT,
            "JSON": SqlType.JSON,
        }[s]


sql_pytype = {
    SqlType.INTEGER: int,
    SqlType.REAL: float,
    SqlType.TEXT: str,
    SqlType.JSON: dict,
}

sql_name = {
    SqlType.INTEGER: "INTEGER",
    SqlType.REAL: "REAL",
    SqlType.TEXT: "TEXT",
    SqlType.JSON: "TEXT",
}

base_model_name = "Model"


def _ast_normalize_rec(node: ast.AST) -> None:
    if not isinstance(node, ast.AST):
        return
    strip = ["end_lineno", "col_offset", "end_col_offset"]
    if hasattr(node, "lineno"):
        setattr(node, "lineno", None)
    for item in strip:
        if hasattr(node, item):
            delattr(node, item)
    for _, v in node.__dict__.items():
        _ast_normalize_rec(v)


def _ast_normalize(f: str) -> ast.AST:
    p = ast.parse(f)
    _ast_normalize_rec(p)
    if isinstance(p, ast.Module):
        if len(p.body) > 1:
            return p.body
        else:
            return p.body[0]
    else:
        return p


def _pytypes(schema: Schema) -> List[ast.stmt]:
    pytypes = {}
    for model in schema.models:
        for field in model.fields:
            if field.pytype is not None:
                if field.pytype not in pytypes:
                    pytypes[field.pytype] = sql_pytype[field.sqltype]
                if pytypes[field.pytype] != sql_pytype[field.sqltype]:
                    raise ValueError(
                        "Mixed-type wrapper",
                        {
                            "pytype": field.pytype,
                            "expected": pytypes[field.pytype],
                            "found": field.sqltype,
                        },
                    )

    pytype_defs = []
    for name, pytype in pytypes.items():
        pytype_defs.append(
            ast.ClassDef(
                name=name,
                bases=[ast.Name(id=pytype.__name__, expr_context_ctx=ast.Load())],
                body=[ast.Pass()],
                decorator_list=[],
                keywords=[],
            )
        )

    return pytype_defs


def _model_init(model: Model, field_annot: Dict[Field, ast.AST]) -> ast.FunctionDef:
    init_args = []
    init_assn = []
    for field in model.fields:
        init_args.append(
            ast.arg(
                arg=field.name,
                annotation=field_annot[field],
                type_comment=None,
            )
        )
        init_assn.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=field.name,
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Name(id=field.name, ctx=ast.Load()),
                type_comment=None,
                lineno=None,
            )
        )

    return ast.FunctionDef(
        name="__init__",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self", annotation=None, type_comment=None), *init_args],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=init_assn,
        decorator_list=[],
        returns=None,
        type_comment=None,
        lineno=None,
    )


def _create_table(model: Model) -> ast.FunctionDef:
    fields_sqlspec = []
    for field in model.fields:
        sqltype = sql_name[field.sqltype]
        if field.pkey:
            sqltype = sqltype + " PRIMARY KEY ASC"
        fields_sqlspec.append(f"{field.name} {sqltype}")
    fields_sql = ", ".join(fields_sqlspec)
    create_stmt = f"CREATE TABLE IF NOT EXISTS {model.name} ({fields_sql})"

    return ast.FunctionDef(
        name="create_table",
        decorator_list=[ast.Name(id="staticmethod", ctx=ast.Load())],
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="cur", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=model.name, ctx=ast.Load()),
                        attr="_cursor",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                ),
                type_comment=None,
                lineno=None,
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="cur", ctx=ast.Load()),
                        attr="execute",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=create_stmt, kind=None)],
                    keywords=[],
                ),
            ),
        ],
        returns=None,
        type_comment=None,
        lineno=None,
    )


def _get_fn(model: Model, field_pytype: Dict[Field, str]) -> ast.FunctionDef:
    pkey = [field for field in model.fields if field.pkey][0]
    select_fields = ", ".join([field.name for field in model.fields])
    select_stmt = f"SELECT {select_fields} FROM {model.name} WHERE {pkey.name} = ?"

    load_targets = []
    jsn_parse = []
    cls_args = []
    for field in model.fields:
        if field.sqltype == SqlType.JSON:
            field_name_txt = field.name + "_txt"
            load_targets.append(ast.Name(id=field_name_txt, ctx=ast.Store()))
            jsn_parse.append(
                ast.Assign(
                    targets=[ast.Name(id=field.name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="json", ctx=ast.Load()),
                            attr="loads",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=field_name_txt, ctx=ast.Load())],
                        keywords=[],
                    ),
                    type_comment=None,
                    lineno=None,
                )
            )
        else:
            load_targets.append(ast.Name(id=field.name, ctx=ast.Store()))

        cls_args.append(ast.Name(id=field.name, ctx=ast.Load()))

    return ast.FunctionDef(
        name="get",
        decorator_list=[ast.Name(id="classmethod", ctx=ast.Load())],
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="cls", annotation=None, type_comment=None),
                ast.arg(arg=pkey.name, annotation=ast.Name(id=field_pytype[pkey], ctx=ast.Load()), type_comment=None),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            _ast_normalize("cur = cls._cursor()"),
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="cur", ctx=ast.Load()), attr="execute", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=select_stmt, kind=None),
                        ast.Tuple(elts=[ast.Name(id=pkey.name, ctx=ast.Load())], ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            ),
            ast.Assign(
                targets=[ast.Tuple(elts=load_targets, ctx=ast.Store())],
                value=_ast_normalize("cur.fetchone()").value,
                type_comment=None,
                lineno=None,
            ),
            *jsn_parse,
            ast.Return(value=ast.Call(func=ast.Name(id="cls", ctx=ast.Load()), args=cls_args, keywords=[])),
        ],
        returns=ast.Constant(value=model.name, kind=None),
        type_comment=None,
        lineno=None,
    )


def _new_fn(model: Model, field_annot: Dict[Field, ast.AST]) -> ast.FunctionDef:
    load_args = []
    cls_args = []
    for field in model.fields:
        if not field.pkey:
            load_args.append(ast.arg(arg=field.name, annotation=field_annot[field], type_comment=None))
            cls_args.append(ast.Name(id=field.name, ctx=ast.Load()))

    return ast.FunctionDef(
        name="new",
        decorator_list=[ast.Name(id="classmethod", ctx=ast.Load())],
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="cls", annotation=None, type_comment=None), *load_args],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id="cls", ctx=ast.Load()),
                    args=[ast.Constant(value=None, kind=None), *cls_args],
                    keywords=[],
                ),
            )
        ],
        returns=ast.Constant(value=model.name, kind=None),
        type_comment=None,
        lineno=None,
    )


def _save_fn(model: Model) -> ast.FunctionDef:
    pkey = [field for field in model.fields if field.pkey][0]

    json_dumps = []
    store_refs = []
    for field in model.fields:
        if field.pkey:
            continue

        store_name = field.name + "_txt"
        if field.sqltype == SqlType.JSON:
            json_dumps.append(
                ast.Assign(
                    targets=[ast.Name(id=store_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="json", ctx=ast.Load()),
                            attr="dumps",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=field.name,
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    ),
                    type_comment=None,
                    lineno=None,
                )
            )
            store_refs.append(ast.Name(id=store_name, ctx=ast.Load()))
        else:
            store_refs.append(
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=field.name,
                    ctx=ast.Load(),
                )
            )

    pkey_ref = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=pkey.name, ctx=ast.Load())

    create_fields = ", ".join([field.name for field in model.fields if not field.pkey])
    q_sql = ", ".join(["?" for field in model.fields if not field.pkey])
    create_stmt = f"INSERT INTO {model.name} ({create_fields}) VALUES ({q_sql})"

    update_fields = ", ".join([f"{field.name} = ?" for field in model.fields if not field.pkey])
    update_stmt = f"UPDATE {model.name} SET {update_fields} WHERE {pkey.name} = ?"

    return ast.FunctionDef(
        name="save",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self", annotation=None, type_comment=None)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            _ast_normalize('cur = self._cursor()'),
            *json_dumps,
            ast.If(
                test=ast.Compare(
                    left=pkey_ref,
                    ops=[ast.Is()],
                    comparators=[ast.Constant(value=None, kind=None)],
                ),
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="cur", ctx=ast.Load()),
                                attr="execute",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Constant(value=create_stmt, kind=None),
                                ast.Tuple(elts=store_refs, ctx=ast.Load()),
                            ],
                            keywords=[],
                        ),
                    ),
                    _ast_normalize("""cur.execute("SELECT last_insert_rowid()")"""),
                    ast.Assign(
                        targets=[
                            ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=pkey.name, ctx=ast.Store())
                        ],
                        value=_ast_normalize("""cur.fetchone()[0]""").value,
                        type_comment=None,
                        lineno=None,
                    ),
                ],
                orelse=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="cur", ctx=ast.Load()),
                                attr="execute",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Constant(value=update_stmt, kind=None),
                                ast.Tuple(elts=[*store_refs, pkey_ref], ctx=ast.Load()),
                            ],
                            keywords=[],
                        ),
                    ),
                ],
            ),
        ],
        decorator_list=[],
        returns=None,
        type_comment=None,
        lineno=None,
    )


def _list_fn(model: Model, field_pytype: dict) -> ast.FunctionDef:
    list_params_args = []
    list_params_names = []
    for field in model.fields:
        if field.sqltype not in {SqlType.JSON} and not field.pkey:
            list_params_args.append(
                ast.arg(
                    arg=field.name,
                    annotation=ast.Subscript(
                        value=ast.Name(id="Optional", ctx=ast.Load()),
                        slice=ast.Name(id=field_pytype[field], ctx=ast.Load()),
                        ctx=ast.Load(),
                    ),
                    type_comment=None,
                )
            )
            list_params_names.append(field.name)

    raw_loads = []
    json_loads = []
    for field in model.fields:
        if field.sqltype == SqlType.JSON:
            raw_name = field.name + "_txt"
            raw_loads.append(ast.Name(id=raw_name, ctx=ast.Store()))
            json_loads.append(
                ast.Assign(
                    targets=[ast.Name(id=field.name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='json', ctx=ast.Load()), attr='loads', ctx=ast.Load()),
                        args=[ast.Name(id=raw_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                    type_comment=None,
                    lineno=None
                )
            )
        else:
            raw_loads.append(ast.Name(id=field.name, ctx=ast.Store()))

    field_names = [field.name for field in model.fields]
    field_names_sql = ", ".join(field_names)
    list_q = f"SELECT {field_names_sql} FROM {model.name} WHERE "

    return ast.FunctionDef(
        name="list",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="cls", annotation=None, type_comment=None), *list_params_args],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[ast.Constant(value=None, kind=None) for _ in range(len(list_params_args))],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="params", ctx=ast.Store())],
                value=ast.Dict(
                    keys=[ast.Constant(value=param, kind=None) for param in list_params_names],
                    values=[ast.Name(id=param, ctx=ast.Load()) for param in list_params_names],
                ),
                type_comment=None,
                lineno=None,
            ),
            _ast_normalize("query_terms = []"),
            _ast_normalize("query_params = []"),
            _ast_normalize(textwrap.dedent('''
            for name, term in params.items():
                if term is not None:
                    query_terms.append(f"{name} = ?")
                    query_params.append(term)
            ''')),
            _ast_normalize(textwrap.dedent('''
            if query_terms:
                query = " AND ".join(query_terms)
            else:
                query = "1"
            ''')),
            ast.Assign(
                targets=[ast.Name(id="list_stmt", ctx=ast.Store())],
                value=ast.JoinedStr(
                    values=[
                        ast.Constant(value=list_q, kind=None),
                        ast.FormattedValue(value=ast.Name(id="query", ctx=ast.Load()), conversion=-1, format_spec=None),
                    ],
                ),
                type_comment=None,
                lineno=None,
            ),
            _ast_normalize("cur = cls._cursor()"),
            _ast_normalize("cur.execute(list_stmt, query_params)"),
            _ast_normalize("results = []"),
            ast.For(
                target=ast.Name(id="row", ctx=ast.Store(), lineno=None),
                iter=_ast_normalize("cur.fetchall()").value,
                body=[
                    ast.Assign(
                        targets=[ast.Tuple(elts=raw_loads, ctx=ast.Store(), lineno=None)],
                        value=ast.Name(id="row", ctx=ast.Load(), lineno=None),
                        type_comment=None,
                        lineno=None,
                    ),
                    *json_loads,
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="results", ctx=ast.Load(), lineno=None),
                                attr="append",
                                ctx=ast.Load(),
                                lineno=None,
                            ),
                            args=[
                                ast.Call(
                                    func=ast.Name(id="cls", ctx=ast.Load(), lineno=None),
                                    args=[ast.Name(id=field.name, ctx=ast.Load()) for field in model.fields],
                                    keywords=[],
                                )
                            ],
                            keywords=[],
                        ),
                    ),
                ],
                orelse=[],
                type_comment=None,
                lineno=None,
            ),
            ast.Return(value=ast.Name(id="results", ctx=ast.Load(), lineno=None)),
        ],
        decorator_list=[ast.Name(id="classmethod", ctx=ast.Load(), lineno=None)],
        returns=ast.List(elts=[ast.Constant(model.name)]),
        type_comment=None,
        lineno=None,
    )


def _to_dict_fn(model: Model) -> ast.FunctionDef:
    keys = []
    values = []
    for field in model.fields:
        keys.append(ast.Constant(value=field.name, kind=None, lineno=None))
        values.append(ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=field.name, ctx=ast.Load()))

    return ast.FunctionDef(
        name='to_dict',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self', annotation=None, type_comment=None, lineno=None)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[ast.Return(value=ast.Dict(keys=keys, values=values))],
        decorator_list=[],
        returns=ast.Name(id='dict', ctx=ast.Load(), lineno=None),
        type_comment=None,
        lineno=None,
    )


def _from_dict_fn(model: Model) -> ast.FunctionDef:
    kw = []
    for field in model.fields:
        kw.append(
            ast.keyword(
                arg=field.name,
                value=ast.Subscript(
                    value=ast.Name(id='d', ctx=ast.Load()),
                    slice=ast.Constant(value=field.name, kind=None),
                    ctx=ast.Load()
                )
            )
        )

    return ast.FunctionDef(
        name='from_dict',
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg='cls', annotation=None, type_comment=None),
                ast.arg(arg='d', annotation=ast.Name(id='dict', ctx=ast.Load()), type_comment=None)
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[ast.Return(value=ast.Call(func=ast.Name(id='cls', ctx=ast.Load()), args=[], keywords=kw))],
        decorator_list=[ast.Name(id='classmethod', ctx=ast.Load())],
        returns=ast.Constant(value=model.name, kind=None),
        type_comment=None,
        lineno=None,
    )


def _update_fn(model: Model, field_annot: Dict[Field, ast.AST]) -> ast.FunctionDef:
    ufields = []
    defaults = []
    assns = []
    for field in model.fields:
        if not field.pkey:
            ufields.append(ast.arg(arg=field.name, annotation=field_annot[field], type_comment=None))
            defaults.append(ast.Constant(value=None, kind=None))
            assns.append(
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=field.name, ctx=ast.Load()),
                        ops=[ast.IsNot()],
                        comparators=[ast.Constant(value=None, kind=None)],
                   ),
                    body=[
                        ast.Assign(
                            targets=[
                                ast.Attribute(
                                    value=ast.Name(id='self', ctx=ast.Load()),
                                    attr=field.name,
                                    ctx=ast.Store()
                                ),
                            ],
                            value=ast.Name(id=field.name, ctx=ast.Load()),
                            type_comment=None,
                            lineno=None,
                        )
                    ],
                    orelse=[],
                )
            )

    return ast.FunctionDef(
        name='update',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self', annotation=None, type_comment=None), *ufields],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=defaults,
        ),
        body=assns,
        decorator_list=[],
        returns=ast.Constant(value=None, kind=None),
        type_comment=None,
        lineno=None,
    )


def build_model_class(model: Model) -> ast.ClassDef:
    field_pytypes = {}
    for field in model.fields:
        if field.pytype is not None:
            field_pytypes[field] = field.pytype
        else:
            field_pytypes[field] = sql_pytype[field.sqltype].__name__

    field_annot = {}
    for field in model.fields:
        pytype = field_pytypes[field]
        if field.nullable or field.pkey:
            field_annot[field] = ast.Subscript(
                value=ast.Name(id="Optional", ctx=ast.Load()),
                slice=ast.Name(id=pytype, ctx=ast.Load()),
                ctx=ast.Load(),
            )
        else:
            field_annot[field] = ast.Name(id=pytype, ctx=ast.Load())

    attrs = [
        ast.Assign(
            targets=[ast.Name(id="TABLE", ctx=ast.Store())],
            value=ast.Constant(value=model.name, kind=None),
            type_comment=None,
            lineno=None,
        ),
    ]

    type_notes = []
    for field in model.fields:
        type_notes.append(
            ast.AnnAssign(
                target=ast.Name(id=field.name, ctx=ast.Store()),
                annotation=field_annot[field],
                value=None,
                simple=1,
            )
        )

    std_methods = [
        _model_init(model, field_annot),
        _create_table(model),
        _get_fn(model, field_pytypes),
        _new_fn(model, field_annot),
        _save_fn(model),
        _list_fn(model, field_pytypes),
        _update_fn(model, field_annot),
        _to_dict_fn(model),
        _from_dict_fn(model),
    ]

    return ast.ClassDef(
        name=model.name,
        bases=[ast.Name(id=base_model_name, ctx=ast.Load())],
        keywords=[],
        body=[*type_notes, *attrs, *std_methods],
        decorator_list=[],
    )


def ormgen(schema: Schema):
    imports = [
        _ast_normalize('import json'),
        _ast_normalize('import sqlite3'),
        _ast_normalize('from typing import Optional, Any'),
        _ast_normalize('from flask import g'),
    ]

    pytypes = _pytypes(schema)
    base_model = ast.ClassDef(
        name=base_model_name,
        bases=[],
        body=[
            ast.FunctionDef(
                name="_cursor",
                args=ast.arguments(
                    args=[],
                    posonlyargs=[],
                    kwonlyargs=[],
                    defaults=[],
                ),
                body=[
                    ast.AnnAssign(
                        target=ast.Name(id="conn", expr_context_ctx=ast.Store()),
                        annotation=ast.Attribute(value=ast.Name(id="sqlite3"), attr="Connection"),
                        value=ast.Attribute(value=ast.Name(id="g"), attr="sql_conn"),
                        simple=None,
                    ),
                    ast.AnnAssign(
                        target=ast.Name(id="cur"),
                        annotation=ast.Attribute(value=ast.Name(id="sqlite3"), attr="Cursor"),
                        value=ast.Call(
                            func=ast.Attribute(value=ast.Name(id="conn"), attr="cursor"),
                            args=[],
                            keywords=[],
                        ),
                        simple=None,
                    ),
                    ast.Return(value=ast.Name(id="cur", ctx=ast.Load())),
                ],
                decorator_list=[ast.Name(id="staticmethod")],
                lineno=None,
                returns=ast.Attribute(value=ast.Name(id="sqlite3"), attr="Cursor"),
                type_comment=None,
            )
        ],
        decorator_list=[],
        keywords=[],
    )

    models = [build_model_class(model) for model in schema.models]

    mod = ast.Module(body=[*imports, *pytypes, base_model, *models], type_ignores=[])
    return ast.unparse(mod)


def main() -> int:
    if len(sys.argv) != 2:
        return 1

    with open(sys.argv[1], "r") as in_f:
        data = json.load(in_f)

    schema = Schema.from_dict(data)
    print(ormgen(schema))


if __name__ == "__main__":
    main()
