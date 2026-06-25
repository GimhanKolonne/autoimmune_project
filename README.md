# Autoimmune ML Project

A machine learning project for analyzing autoimmune disease data.

## Project Structure

```
autoimmune_ml_project/
├── data/                   # Data files
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── config.py         # Configuration settings
│   ├── load_data.py      # Data loading utilities
│   ├── preprocess.py     # Data preprocessing
│   ├── features.py       # Feature engineering
│   ├── train_models.py   # Model training
│   ├── ensemble.py       # Ensemble methods
│   ├── evaluate.py       # Model evaluation
│   └── main.py           # Main execution script
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:

   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```bash
python src/main.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
