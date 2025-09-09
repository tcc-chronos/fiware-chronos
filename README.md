### Criação máquina Virtual WSL

```
wsl --set-default-version 2
wsl --install -d Ubuntu
wsl --set-default Ubuntu
```

### Configuração da WSL

```
# Atualizar o sistema
sudo apt update && sudo apt -y upgrade

# Ferramentas necessárias
sudo apt -y install git make build-essential python3 python3-venv python3-pip
```


### Clonar e configurar o repositório

```
# Clonar e acessar o repositório
git clone https://github.com/tcc-chronos/fiware-chronos.git
cd fiware-chronos

# Criar ambiente Python
python3 -m venv .venv
source .venv/bin/activate

# Instalar as dependências
pip install -r requirements.txt

# Habilitar os scripts
chmod +x scripts/*.sh

# Configurar o pre-commit
pre-commit install
pre-commit run --all-files
```


### Comandos disponíveis

```
make dev-up   # sobe docker compose
make run      # roda uvicorn local com env
make format   # black + isort
make lint     # flake8 + mypy
```
