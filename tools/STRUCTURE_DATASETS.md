# YOLO

### without division into training and validation sets

```
dataset_roo/
            images/
                image_0001.jpg
                image_0002.jpg
                image_0003.jpg
            labels/
                image_0001.txt
                image_0002.txt
                image_0003.txt
            classes.txt
```

### with division into training and validation sets

```
dataset_roo/
    images/
        train/
            image_0001.jpg
            image_0002.jpg
            image_0003.jpg
        val/
            image_0006.jpg
            image_0009.jpg
            image_00020.jpg
    labels/
        train/
            image_0001.txt
            image_0002.txt
            image_0003.txt
        val/
            image_0006.txt
            image_0009.txt
            image_00020.txt
    classes.txt
```

# COCO

```
dataset_roo/
    images/
        image1.jpg
        image2.jpg
    annotations.json
```