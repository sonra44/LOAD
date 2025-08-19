"""Простейшие типы, имитирующие protobuf common_types."""
from dataclasses import dataclass


@dataclass
class UUID:
    """Минималистичная реализация protobuf UUID"""
    value: str = ""

    def CopyFrom(self, other: "UUID") -> None:
        self.value = other.value

