# Instructions for creating a dataset via json

Run
```bash
uv run annotate_json.py --video-path path_to_video --json-path path_to_json --type-save yolo
```

## An example of a `json` file with different annotation modes and classes
```json
[
    {
        "class_name": "raccon",
        "prompt": {
            "mode": "point",
            "point_coords": [[531, 230]],
            "point_labels": [1]
        }
    },
    {
        "class_name": "cat",
        "prompt": {
            "mode": "point",
            "point_coords": [
                [45, 321],
                [226, 360],
                [194, 313]
            ],
            "point_labels": [1, 1, 1]
        }
    }
]
```

## Possible annotation modes

### Through a point
```json
[
    {
        "class_name": "cat",
        "prompt": {
            "mode": "point",
            "point_coords": [
                [45, 321],
                [226, 360],
                [194, 313]
            ],
            "point_labels": [1, 1, 1]
        }
    }
]
```

If you need to point to a specific part of the object, you must additionally specify the coordinates of these points in another array of `point_coords` and `point_labels` as in the example below.
```json
[
    {
        "class_name": "cat",
        "prompt": {
            "mode": "point",
            "point_coords": [
                [45, 321],
                [
                    [226, 360],
                    [194, 313]
                ]
            ],
            "point_labels": [1, [1, 1]]
        }
    }
]
```

### Via bbox
The coordinates of the frame are set in the format `x, y, x+w, y+h`
```json
[
    {
        "class_name": "raccon",
        "prompt": {
            "mode": "box",
            "boxes": [[476, 166, 578, 320], [8, 252, 99, 401], [106, 335, 317, 425]]
        }
    }
]
```

### Combined mode `both`
```json
[
{
    "class_name": "raccon",
    "prompt": {
        "mode": "both",
        "point_coords": [[575, 750]],
        "point_labels": [0],
        "boxes": [[425, 600, 700, 875]],
    }
}
]
```
