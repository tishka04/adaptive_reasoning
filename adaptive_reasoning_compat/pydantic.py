"""Tiny Pydantic-compatible subset used for local test environments.

This shim is intentionally small: it supports only the features this repo
uses in tests and CPU-only smoke runs.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union, get_args, get_origin, get_type_hints

try:
    from types import UnionType
except ImportError:  # pragma: no cover - Python < 3.10
    UnionType = ()  # type: ignore[assignment]


_UNSET = object()


class ValidationError(ValueError):
    """Minimal validation error type."""


@dataclass
class PrivateAttrInfo:
    default: Any = _UNSET
    default_factory: Callable[[], Any] | None = None

    def get_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is _UNSET:
            return None
        return self.default


@dataclass
class FieldInfo:
    default: Any = _UNSET
    default_factory: Callable[[], Any] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is _UNSET:
            raise ValidationError("Required field missing")
        return self.default


def Field(
    default: Any = _UNSET,
    *,
    default_factory: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> FieldInfo:
    """Capture field defaults and schema metadata."""
    if default is Ellipsis:
        default = _UNSET
    return FieldInfo(default=default, default_factory=default_factory, metadata=dict(kwargs))


def PrivateAttr(
    default: Any = _UNSET,
    *,
    default_factory: Callable[[], Any] | None = None,
) -> PrivateAttrInfo:
    """Capture runtime-only private attributes."""
    return PrivateAttrInfo(default=default, default_factory=default_factory)


def model_validator(*, mode: str = "after") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator placeholder for after-model validators."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "__compat_model_validator_mode__", mode)
        return func

    return decorator


def field_validator(*field_names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator placeholder for field validators."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        target = func.__func__ if isinstance(func, (classmethod, staticmethod)) else func
        setattr(target, "__compat_field_validator_fields__", tuple(field_names))
        return func

    return decorator


class _ComputedProperty(property):
    """Property marker used by `computed_field`."""


def computed_field(**_kwargs: Any) -> Callable[[Any], Any]:
    """Decorator placeholder for computed-field properties."""

    def decorator(obj: Any) -> Any:
        if isinstance(obj, property):
            return _ComputedProperty(obj.fget, obj.fset, obj.fdel, obj.__doc__)
        return _ComputedProperty(obj)

    return decorator


class BaseModel:
    """Very small subset of `pydantic.BaseModel`."""

    __fields__: Dict[str, FieldInfo] = {}
    __field_types__: Dict[str, Any] = {}
    __validators__: List[Callable[..., Any]] = []
    __field_validators__: Dict[str, List[Callable[..., Any]]] = {}
    __private_attrs__: Dict[str, PrivateAttrInfo] = {}
    __computed_fields__: List[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        fields: Dict[str, FieldInfo] = {}
        field_types: Dict[str, Any] = {}
        validators: List[Callable[..., Any]] = []
        field_validators: Dict[str, List[Callable[..., Any]]] = {}
        private_attrs: Dict[str, PrivateAttrInfo] = {}
        computed_fields: List[str] = []

        for base in reversed(cls.__mro__[1:]):
            fields.update(getattr(base, "__fields__", {}))
            field_types.update(getattr(base, "__field_types__", {}))
            validators.extend(getattr(base, "__validators__", []))
            for key, items in getattr(base, "__field_validators__", {}).items():
                field_validators.setdefault(key, []).extend(items)
            private_attrs.update(getattr(base, "__private_attrs__", {}))
            computed_fields.extend(getattr(base, "__computed_fields__", []))

        hints = get_type_hints(cls, include_extras=True)
        for name, annotation in hints.items():
            default = cls.__dict__.get(name, _UNSET)
            if name.startswith("_"):
                if isinstance(default, PrivateAttrInfo):
                    private_attrs[name] = default
                continue
            if isinstance(default, FieldInfo):
                field_info = default
            elif default is _UNSET:
                field_info = Field()
            else:
                field_info = Field(default=default)
            fields[name] = field_info
            field_types[name] = annotation

        for name, attr in cls.__dict__.items():
            if isinstance(attr, _ComputedProperty):
                computed_fields.append(name)

            target = attr
            if isinstance(attr, (classmethod, staticmethod)):
                target = attr.__func__

            if callable(target) and getattr(target, "__compat_model_validator_mode__", None) == "after":
                validators.append(getattr(cls, name))

            fields_for_validator = getattr(target, "__compat_field_validator_fields__", None)
            if callable(target) and fields_for_validator:
                for field_name in fields_for_validator:
                    field_validators.setdefault(field_name, []).append(getattr(cls, name))

        cls.__fields__ = fields
        cls.__field_types__ = field_types
        cls.__validators__ = validators
        cls.__field_validators__ = field_validators
        cls.__private_attrs__ = private_attrs
        cls.__computed_fields__ = list(dict.fromkeys(computed_fields))

    def __init__(self, **data: Any) -> None:
        remaining = dict(data)
        for name, annotation in self.__field_types__.items():
            if name in remaining:
                raw_value = remaining.pop(name)
            else:
                raw_value = self.__fields__[name].get_default()
            try:
                value = _coerce_value(annotation, raw_value)
            except ValidationError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise ValidationError(f"{self.__class__.__name__}.{name}: {exc}") from exc
            setattr(self, name, value)

        for name, validators in self.__field_validators__.items():
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            for validator in validators:
                value = _invoke_validator(validator, self.__class__, value)
            setattr(self, name, value)

        for name, attr in self.__private_attrs__.items():
            setattr(self, name, attr.get_default())

        for key, value in remaining.items():
            setattr(self, key, value)

        for validator in self.__validators__:
            result = validator(self)
            if isinstance(result, self.__class__) and result is not self:
                self.__dict__.update(result.__dict__)

    def model_dump(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in self.__field_types__:
            info = self.__fields__[name]
            if info.metadata.get("exclude"):
                continue
            out[name] = _dump_value(getattr(self, name))
        for name in self.__computed_fields__:
            out[name] = _dump_value(getattr(self, name))
        return out

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        exclude_none: bool = False,
        **_kwargs: Any,
    ) -> str:
        payload = self.model_dump()
        if exclude_none:
            payload = {k: v for k, v in payload.items() if v is not None}
        return json.dumps(payload, indent=indent, default=_json_default)

    @classmethod
    def model_validate(cls, data: Any) -> "BaseModel":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValidationError(f"{cls.__name__} expects a dict payload")
        return cls(**data)

    @classmethod
    def model_validate_json(cls, text: str | bytes | bytearray, **_kwargs: Any) -> "BaseModel":
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")
        return cls.model_validate(json.loads(text))

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for name, annotation in cls.__field_types__.items():
            info = cls.__fields__[name]
            schema = _schema_for_type(annotation)
            schema.update(info.metadata)
            properties[name] = schema
            if info.default is _UNSET and info.default_factory is None:
                required.append(name)
        out: Dict[str, Any] = {
            "title": cls.__name__,
            "type": "object",
            "properties": properties,
        }
        if required:
            out["required"] = required
        return out

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        args = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__field_types__)
        return f"{self.__class__.__name__}({args})"


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    if annotation in (Any, object):
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if value is None:
            return None
        for candidate in non_none:
            try:
                return _coerce_value(candidate, value)
            except Exception:
                continue
        return value

    if origin is list:
        inner = args[0] if args else Any
        return [_coerce_value(inner, item) for item in (value or [])]

    if origin is dict:
        key_t, val_t = args if len(args) == 2 else (Any, Any)
        return {
            _coerce_value(key_t, key): _coerce_value(val_t, val)
            for key, val in dict(value).items()
        }

    if origin is tuple:
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(args[0], item) for item in items)
        return tuple(
            _coerce_value(args[idx] if idx < len(args) else Any, item)
            for idx, item in enumerate(items)
        )

    if str(origin).endswith("Literal"):
        return value

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            if isinstance(value, annotation):
                return value
            if not isinstance(value, dict):
                raise ValidationError(f"Expected mapping for nested {annotation.__name__}")
            return annotation.model_validate(value)
        if issubclass(annotation, Enum):
            if isinstance(value, annotation):
                return value
            return annotation(value)
        if annotation is bool:
            if isinstance(value, str):
                return value.strip().lower() not in {"", "0", "false", "no", "off"}
            return bool(value)
        if annotation in (int, float, str):
            if isinstance(value, annotation):
                return value
            return annotation(value)

    return value


def _invoke_validator(validator: Callable[..., Any], cls: type, value: Any) -> Any:
    if isinstance(validator, types.MethodType):
        return validator(value)
    return validator(cls, value)


def _dump_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    if isinstance(value, tuple):
        return [_dump_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump_value(val) for key, val in value.items()}
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _schema_for_type(annotation: Any) -> Dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        schema = _schema_for_type(non_none[0] if non_none else Any)
        if len(non_none) != len(args):
            schema["nullable"] = True
        return schema

    if origin is list:
        return {"type": "array", "items": _schema_for_type(args[0] if args else Any)}

    if origin is dict:
        return {"type": "object"}

    if origin is tuple:
        return {"type": "array"}

    if str(origin).endswith("Literal"):
        return {"enum": list(args)}

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return {"type": "string", "enum": [item.value for item in annotation]}
        if issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}

    return {}
