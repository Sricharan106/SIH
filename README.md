## Run NetOracle

Create the environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the local temporal Transformer:

```bash
python training/train_world_model.py
```

Start the dashboard:

```bash
streamlit run app.py
```

Upload a headered or headerless UNSW-NB15 CSV, or a flow CSV containing common fields such as bytes, duration, packets, or destination port. The dashboard reports `Normal` or `Suspicious`, a risk score, the decision threshold, and suspicious windows.
