from typing import Optional

import cv2
import numpy as np


class FrameViewer:
    WINDOW_NAME: str = 'Frame Viewer'
    CONTROL_PANEL_HEIGHT: int = 40
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_INFO: float = 0.6
    FONT_SCALE_CONTROLS: float = 0.45
    FONT_THICKNESS: int = 1
    POINT_RADIUS: int = 5
    POINT_COLOR: tuple[int, int, int] = (0, 0, 255)  # BGR (красный)
    GRID_COLOR: tuple[int, int, int] = (0, 255, 0)  # BGR (зелёный)

    def __init__(self, window_name: str = WINDOW_NAME):
        self.window_name = window_name
        self.current_frame: Optional[np.ndarray] = None
        self.current_points: list[tuple[int, int]] = []

    def show_frame(
        self,
        frame: np.ndarray,
        points: Optional[list[tuple[int, int]]] = None,
        frame_index: int = 0,
        total_frames: int = 0,
    ) -> int:
        """
        Показ кадра с опциональными точками и элементами управления.

        Args:
            frame: Кадр для отображения.
            points: Список точек для отображения.
            frame_index: Индекс текущего кадра.
            total_frames: Общее количество кадров.
            show_controls: Показывать панель управления.

        Returns:
            int: Нажатая клавиша (или 0).
        """
        self.current_frame = frame.copy()
        self.current_points = points or []

        self._draw_control_panel(frame_index, total_frames)

        self._draw_points()
        self._draw_grid()

        cv2.imshow(self.window_name, self.current_frame)
        return cv2.waitKey(1)

    def _draw_control_panel(self, frame_index: int, total_frames: int) -> None:
        """Рисование панели управления."""
        if self.current_frame is None:
            return

        h, w = self.current_frame.shape[:2]

        # Фон панели
        cv2.rectangle(
            self.current_frame,
            (0, 0),
            (w, self.CONTROL_PANEL_HEIGHT),
            (0, 0, 0),
            -1,
        )

        # Информация о кадре
        cv2.putText(
            self.current_frame,
            f'Frame {frame_index} / {total_frames}',
            (10, 20),
            self.FONT,
            self.FONT_SCALE_INFO,
            (255, 255, 255),
            self.FONT_THICKNESS,
        )

        # Элементы управления
        controls = '[S] Save [W] Empty [D] Next [A] Back [Q] Quit'
        cv2.putText(
            self.current_frame,
            controls,
            (10, 35),
            self.FONT,
            self.FONT_SCALE_CONTROLS,
            (255, 255, 255),
            self.FONT_THICKNESS,
        )

    def _draw_points(self) -> None:
        """Рисование точек."""
        if self.current_frame is None:
            return
        for x, y in self.current_points:
            cv2.circle(
                self.current_frame,
                (x, y),
                self.POINT_RADIUS,
                self.POINT_COLOR,
                -1,
            )

    def _draw_grid(self) -> None:
        """Рисование сетки."""
        if self.current_frame is None:
            return

        h, w = self.current_frame.shape[:2]

        # Вертикальная линия
        cv2.line(
            self.current_frame,
            (w // 2, self.CONTROL_PANEL_HEIGHT),
            (w // 2, h),
            self.GRID_COLOR,
            1,
        )

        # Горизонтальная линия
        cv2.line(
            self.current_frame,
            (0, h // 2),
            (w, h // 2),
            self.GRID_COLOR,
            1,
        )

    def clear_points(self) -> None:
        """Очистка всех точек."""
        self.current_points.clear()

    def get_points(self) -> list[tuple[int, int]]:
        """Получение текущего списка точек."""
        return self.current_points.copy()

    def close(self) -> None:
        """Закрытие окна."""
        cv2.destroyAllWindows()

    @staticmethod
    def set_mouse_callback(
        window_name: str,
        callback,
        param: Optional[object] = None,
    ) -> None:
        """
        Установка обработчика мыши.

        Args:
            window_name: Имя окна.
            callback: Функция обратного вызова.
            param: Дополнительный параметр.
        """
        # Создаём окно если оно ещё не существует
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, callback, param)
