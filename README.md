# Track Anything Annotate

<div align="center">
  <a href="https://arxiv.org/abs/2505.17884"> 
    <img src="https://img.shields.io/badge/📄-Arxiv_2505.17884-red.svg?style=flat-square" alt="Paper on arXiv">
  </a>
  <a href="https://huggingface.co/spaces/lniki/track-anything-annotate">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face_Space-informational.svg?style=flat-square" alt="Open in Spaces">
  </a>
</div>

***Track Anything Annotate*** is a flexible tool for tracking, segmentation, and annotation of videos. It allows creating datasets from videos in YOLO and COCO formats. It is based on Segment Anything 2 and allows specifying any objects for tracking and segmentation.

Read this in other languages: [English](README.md) | [Русский](README.ru.md)
## Quick Start

### 🛠️ Installation via `uv`

```bash
# For CUDA  
uv sync --extra cu129

# For CPU  
uv sync --extra cpu
```

### Download Models

```bash
uv run checkpoints/download_models.py
```

### Run the Demo (Access at http://127.0.0.1:8080 )

```bash
gradio demo.py
```

![alt text](video-test/cache/image.png)

### Dataset Creation

**Currently, it only works with one class.**

Type of saving
- yolo
- coco

```bash
uv run annotation.py --video-path path_to_video --names-class name_class --type-save yolo
```

---

### Installation via `pip`

#### Install for CUDA Windows
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

#### Install Linux
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

## 🗺️ Roadmap and Improvements

*   [x] Tracking single class export in YOLO.
*   [x] **New export formats:** Adding support for COCO JSON.
*   [ ] **New export formats:** Adding support for Pascal VOC XML.      
*   [ ] **Multi-class annotation:** Ability to track multiple different classes.
*   [ ] **Image annotation:** Ability to collect and annotate your own dataset based on images.

## 📚 Citation 

If you use this project in your work, please cite the paper: 

```bibtex
@article{ivanov2025track,
    title={Track Anything Annotate: Video annotation and dataset generation of computer vision models},
    author={Ivanov, Nikita and Klimov, Mark and Glukhikh, Dmitry and Chernysheva, Tatiana and Glukhikh, Igor},
    journal={arXiv preprint arXiv:2505.17884},
    year={2025}
}
```
