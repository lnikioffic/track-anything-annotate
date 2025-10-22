# Track Anything Annotate

<div align="center">
  <a href="https://arxiv.org/abs/2505.17884"> 
    <img src="https://img.shields.io/badge/📄-Arxiv_2505.17884-red.svg?style=flat-square" alt="Paper on arXiv">
  </a>
  <a href="https://huggingface.co/spaces/lniki/track-anything-annotate">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face_Space-informational.svg?style=flat-square" alt="Open in Spaces">
  </a>
</div>

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

```bash
uv run annotation.py
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
