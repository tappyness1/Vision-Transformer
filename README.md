# Vision Transformer

## What this is about

Just a simple implementation based on the Vision Tranformer paper to understand what the fuss is all about for Transformers without needing to leave the safety of Computer Vision.

## How to run

Make sure you change the directory of your data. CIFAR10 is a good dataset try out - https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html

```
python -m src.main
```

Run the app:

```
streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false &
```

## Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch.txt
pip install -e .
```