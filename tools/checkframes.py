import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw(images_path, labels_path):
    pa = Path.cwd() / 'video-test' / 'dt-raccon-cat'
    pal = Path(pa / labels_path)
    dir_list = sorted(pal.glob('*'))
    for each in dir_list:
        draw_labels_on_image(
            pa, images_path, pal, os.path.splitext(os.path.split(each)[1])[0]
        )


def draw_labels_on_image(path, images_path, labels_path, file_name) -> None:
    """Draws labels on an image using the provided labels file."""
    image = Image.open(f'{path}/{images_path}/{file_name}.jpg')
    width, height = image.size

    with open(labels_path / f'{file_name}.txt') as file:
        for line in file:
            class_id, x_center, y_center, width_factor, height_factor = line.split()
            x = float(x_center) * width
            y = float(y_center) * height
            width_factor = int(float(width_factor) * width)
            height_factor = int(float(height_factor) * height)

            font = ImageFont.truetype('arial.ttf', 15)
            draw = ImageDraw.Draw(image)
            draw.text((x + 50, y + 50), class_id, font=font, fill=(255, 0, 0))
            draw.rectangle(
                (
                    (x - width_factor // 2),
                    (y - height_factor // 2),
                    (x + width_factor // 2),
                    (y + height_factor // 2),
                ),
                outline=(0, 255, 0),
                width=3,
            )

    image.save(f'{path}/t{file_name}.jpg', quality=95)


if __name__ == '__main__':
    draw('images', 'labels')

    # # Укажите путь к папке, которую нужно архивировать
    # folder_path = Path.cwd() / 'data' / 'Helmetxy'

    # # Укажите путь и имя архива
    # archive_path = Path.cwd() / 'data' / 'Helmetxy'

    # # Создайте архив
    # shutil.make_archive(archive_path, 'zip', folder_path)
