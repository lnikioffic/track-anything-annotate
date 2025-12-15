from typing import TypedDict


class PointPrompt(TypedDict):
    mode: str
    point_coords: list[list[int] | list[list[int]]]
    point_labels: list[int | list[int]]


class BoxPrompt(TypedDict):
    mode: str
    boxes: list[list[int] | list[list[int]]]


class BothPrompt(TypedDict):
    mode: str
    point_coords: list[list[int] | list[list[int]]]
    point_labels: list[int | list[int]]
    boxes: list[list[int] | list[list[int]]]


Prompt = PointPrompt | BoxPrompt | BothPrompt


class AnnotationItem(TypedDict):
    class_name: str
    prompt: Prompt


class AnnotationInfo(TypedDict):
    frames_path: list[str]
    keypoints: dict[int, list[tuple[int, int]]]
