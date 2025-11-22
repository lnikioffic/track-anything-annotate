import cv2
import numpy as np

from segmenter import Segmenter
from tools.mask_display import visualize_unique_mask
from tools.mask_merge import merge_masks
from XMem2.inference.interact.interactive_utils import overlay_davis
from tools.types import Prompt, PointPrompt


class SegmenterController:
    def __init__(self):
        """
        Инициализация контроллера для работы с Segmenter2.
        :param device: Устройство для выполнения вычислений ('cuda' или 'cpu').
        """
        self.segmenter = Segmenter()
        self.image_set = False

    def load_image(self, image: np.ndarray):
        """
        Загружает изображение в Segmenter2.
        :param image: Изображение в формате NumPy массива (H, W, C).
        """
        if self.image_set:
            print('Изображение уже загружено. Сбросьте его перед загрузкой нового.')
            return
        try:
            self.segmenter.set_image(image)
            self.image_set = True
            print('Изображение успешно загружено.')
        except Exception as e:
            print(f'Ошибка при загрузке изображения: {e}')

    def reset_image(self):
        """
        Сбрасывает текущее изображение в Segmenter2.
        """
        if not self.image_set:
            print('Нет загруженного изображения для сброса.')
            return
        try:
            self.segmenter.reset_image()
            self.image_set = False
            print('Изображение успешно сброшено.')
        except Exception as e:
            print(f'Ошибка при сбросе изображения: {e}')

    def _process_point_prompt(
        self,
        point_coords: list[list[int] | list[list[int]]],
        point_labels: list[int | list[int]],
    ) -> list[dict[str, np.ndarray]]:
        """
        Обрабатывает промпт для точек.
        :param point_coords: Координаты точек.
        :param point_labels: Метки точек.
        :return: Список словарей с подготовленными данными для predict.
        """
        prompts = []
        for coords, labels in zip(point_coords, point_labels):
            # Определяем, является ли текущий элемент списком координат или одной координатой
            if isinstance(coords[0], list) and isinstance(labels, list):
                # Если несколько точек и меток, multimask=False
                prompt = {
                    'point_coords': np.array(coords),
                    'point_labels': np.array(labels),
                }
                prompts.append((prompt, False))
            else:
                # Если одна точка, multimask=True
                prompt = {
                    'point_coords': np.array([coords]),
                    'point_labels': np.array([labels]),
                }
                prompts.append((prompt, True))
        return prompts

    def _process_box_prompt(
        self, boxes: list[list[int]]
    ) -> list[dict[str, np.ndarray]]:
        """
        Обрабатывает промпт для рамок.
        :param boxes: Рамки.
        :return: Список словарей с подготовленными данными для predict.
        """
        prompts = []
        for box in boxes:
            prompt = {'boxes': np.array([box])}
            prompts.append((prompt, True))  # multimask=True для каждой рамки
        return prompts

    def _process_both_prompt(
        self,
        point_coords: list[list[int] | list[list[int]] | None],
        point_labels: list[int | list[int] | None],
        boxes: list[list[int]],
    ) -> list[dict[str, np.ndarray]]:
        """
        Обрабатывает промпт для комбинированного режима.
        :param point_coords: Координаты точек.
        :param point_labels: Метки точек.
        :param boxes: Рамки.
        :return: Список словарей с подготовленными данными для predict.
        """
        prompts = []
        for box, coords, labels in zip(boxes, point_coords, point_labels):
            prompt = {'boxes': np.array([box])}
            if coords is not None and labels is not None:
                prompt['point_coords'] = np.array([coords])
                prompt['point_labels'] = np.array([labels])
                prompts.append((prompt, False))  # multimask=False, если есть точки
            else:
                prompts.append((prompt, True))  # multimask=True, если точек нет
        return prompts

    def predict_from_prompts(
        self, prompts: Prompt
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Выполняет предсказание на основе заданного промпта.
        :param prompts: Словарь с данными для предсказания.
        :return: Список кортежей (маски, оценки, логиты).
        """
        if not self.image_set:
            raise RuntimeError('Изображение не загружено. Сначала вызовите load_image.')

        mode = prompts.get('mode')
        results = []

        if mode == 'point':
            point_coords = prompts.get('point_coords', [])
            point_labels = prompts.get('point_labels', [])
            processed_prompts = self._process_point_prompt(point_coords, point_labels)
        elif mode == 'box':
            boxes = prompts.get('boxes', [])
            processed_prompts = self._process_box_prompt(boxes)
        elif mode == 'both':
            point_coords = prompts.get(
                'point_coords', [None] * len(prompts.get('boxes', []))
            )
            point_labels = prompts.get(
                'point_labels', [None] * len(prompts.get('boxes', []))
            )
            boxes = prompts.get('boxes', [])
            processed_prompts = self._process_both_prompt(
                point_coords, point_labels, boxes
            )
        else:
            raise ValueError("Режим должен быть 'point', 'box' или 'both'.")

        # TODO: добавить вариант без цикла

        for prompt, multimask in processed_prompts:
            try:
                masks, scores, logits = self.segmenter.predict(
                    prompt, mode=mode, multimask=multimask
                )
                results.append((masks, scores, logits))
            except Exception as e:
                print(f'Ошибка при выполнении предсказания: {e}')
                raise

        return results


if __name__ == '__main__':
    # Создаем контроллер
    controller = SegmenterController()

    path = 'video-test/truck.jpg'
    path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()
    controller.load_image(frame)
    import timeit

    # Пример 1: Точки
    prompts: PointPrompt = {
        'mode': 'point',
        'point_coords': [[[531, 230], [45, 321]], [226, 360], [194, 313]],
        'point_labels': [[1, 1], 1, 1],
    }

    # prompts = {
    #     'mode': 'point',
    #     'point_coords': [[[531, 230], [45, 321]], [226, 360], [194, 313]],
    #     'point_labels': [[1, 0], 1, 1],
    # }

    def run_segmentation():
        prompts: PointPrompt = {
            'mode': 'point',
            'point_coords': [[531, 230], [45, 321], [226, 360], [194, 313]],
            'point_labels': [1, 0, 1, 1],
        }
        return controller.predict_from_prompts(prompts)

    results = controller.predict_from_prompts(prompts)

    execution_time_ms = timeit.timeit(run_segmentation, number=1) * 1000
    print(f'Время выполнения: {execution_time_ms:.2f} мс')
    # Пример 2: Рамки
    # prompts = {
    #     'mode': 'box',
    #     'boxes': [
    #         [476, 166, 578, 320],
    #         [8, 252, 99, 401],
    #         [106, 335, 317, 425],
    #         [155, 283, 225, 339],
    #     ],
    # }
    # results = controller.predict_from_prompts(prompts)

    # Пример 3: Комбинированный режим
    # prompts = {
    #     'mode': 'both',
    #     'point_coords': [[575, 750]],
    #     'point_labels': [0],
    #     'boxes': [[425, 600, 700, 875]],
    # }
    # results = controller.predict_from_prompts(prompts)

    print(len(results))
    res = [result[np.argmax(scores)] for result, scores, logits in results]
    mask, unique_mask = merge_masks(res)
    f = overlay_davis(frame, unique_mask)
    mask = visualize_unique_mask(unique_mask)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cv2.imshow('asd', mask)
    cv2.imshow('asd', f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Сбрасываем изображение
    controller.reset_image()
