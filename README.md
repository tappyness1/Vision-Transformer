# Vision Transformer

## What this is about

Just a simple implementation based on the Vision Tranformer paper to understand what the fuss is all about for Transformers without needing to leave the safety of Computer Vision.

## How to run

Make sure you change the directory of your data. I used Flowers102 dataset hence 102 classes.

```
python -m src.main
```

Run the app:

```
streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false &
```
