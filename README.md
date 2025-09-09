```
python -m venv venv
. .venv/Scripts/activate
pip install -r requirements.txt

chmod +x scripts/*.sh
```

```
python -m pre_commit install
python -m pre_commit run --all-files
```

```
make dev-up   # sobe docker compose
make run      # roda uvicorn local com env
make format   # black + isort
make lint     # flake8 + mypy
```
