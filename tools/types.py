from typing import TypedDict


class PointPrompt(TypedDict):
    mode: str
    point_coords: list[list[int] | list[list[int]]]
    point_labels: list[int | list[int]]


class BoxPrompt(TypedDict):
    mode: str
    box_coords: list[list[int] | list[list[int]]]


class BothPrompt(TypedDict):
    mode: str
    point_coords: list[list[int] | list[list[int]]]
    point_labels: list[int | list[int]]
    box_coords: list[list[int] | list[list[int]]]
    box_labels: list[int | list[int]]


Prompt = PointPrompt | BoxPrompt | BothPrompt
