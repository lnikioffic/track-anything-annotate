# Track Anything Annotate

<div align="center">
  <a href="https://arxiv.org/abs/2505.17884"> 
    <img src="https://img.shields.io/badge/📄-Arxiv_2505.17884-red.svg?style=flat-square" alt="Paper on arXiv">
  </a>
  <a href="https://huggingface.co/spaces/lniki/track-anything-annotate">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face_Space-informational.svg?style=flat-square" alt="Open in Spaces">
  </a>
</div>

***Track Anything Annotate*** это гибкий инструмент для отслеживания, сегментации и аннотирования видео. Он позволяет создавать датасеты из видео в формате YOLO и COCO. Он разработаен на основе Segment Anything 2 и позволяет указать любые объекты для отслеживания и сегментации.

Читайте это на других языках: [English](README.md) | [Русский](README.ru.md)
## Быстрый старт

### 🛠️ Установка через `uv`

```bash
# For CUDA  
uv sync --extra cu129

# For CPU  
uv sync --extra cpu
```

### Скачать модели

```bash
uv run checkpoints/download_models.py
```

### Запустить Демо (Доступ по адресу http://127.0.0.1:8080 )

```bash
gradio demo.py
```

![alt text](video-test/cache/image.png)

### Создание Dataset

**Текущая версия работает только с одним классом.**

Тип сохранения
- yolo
- coco

```bash
uv run annotation.py --video-path path_to_video --names-class name_class --type-save yolo
```

---

### Установка через `pip`

#### Установка для CUDA Windows
```bash
# Clone the repository:
git clone https://github.com/lnikioffic/track-anything-annotate.git
cd track-anything-annotate

# Install dependencies:
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu129

# Download Models
python checkpoints/download_models.py

# Run the gradio demo.
python demo.py

# Dataset Creation
python annotation.py
```

#### Установка для Linux
```bash
# Clone the repository:
git clone https://github.com/lnikioffic/track-anything-annotate.git
cd track-anything-annotate

# Install dependencies:
pip install -r requirements.txt

# Download Models
python checkpoints/download_models.py

# Run the gradio demo.
python demo.py

# Dataset Creation
python annotation.py
```

## 🗺️ Roadmap и Планы

Основываясь на обратной связи от пользователей (CustDev), мы планируем следующие улучшения:

*   [x] Трекинг одного объекта и экспорт в YOLO.
*   [x] **Новые форматы экспорта:** Добавление поддержки COCO JSON.
*   [ ] **Новые форматы экспорта:** Добавление поддержки Pascal VOC XML.      
*   [ ] **Мульти-классовая разметка:** Возможность отслеживать несколько разных объектов.
*   [ ] **Разметка изображений:** Возможность собрать и разметить свой датасет на основе изображений.

## 📚 Цитирование 

Если вы используете этот проект в своей работе, пожалуйста, цитируйте статью: 

```bibtex
@article{ivanov2025track,
    title={Track Anything Annotate: Video annotation and dataset generation of computer vision models},
    author={Ivanov, Nikita and Klimov, Mark and Glukhikh, Dmitry and Chernysheva, Tatiana and Glukhikh, Igor},
    journal={arXiv preprint arXiv:2505.17884},
    year={2025}
}
```
