# FIWARE Chronos

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.14-green.svg)](https://www.mongodb.com/)

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Arquitetura](#-arquitetura)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Instalação e Configuração](#-instalação-e-configuração)
  - [Configuração do WSL (para usuários Windows)](#configuração-do-wsl-para-usuários-windows)
  - [Configuração do Ambiente de Desenvolvimento](#configuração-do-ambiente-de-desenvolvimento)
  - [Configuração do Docker](#configuração-do-docker)
- [API Endpoints](#-api-endpoints)
- [Modelos de Dados](#-modelos-de-dados)
- [Desenvolvimento](#-desenvolvimento)
- [Comandos Úteis](#-comandos-úteis)
- [Licença](#-licença)

## 🚀 Sobre o Projeto

**FIWARE Chronos** é um Generic Enabler (GE) para treinamento e implantação de modelos de deep learning integrado com a plataforma FIWARE. O projeto permite gerenciar configurações de modelos, treinar modelos com séries temporais provenientes do FIWARE STH-Comet, e realizar previsões usando os modelos treinados.

O sistema é projetado para facilitar a integração de soluções de machine learning com ecossistemas IoT baseados em FIWARE, permitindo:

- Gerenciamento completo de configurações de modelos de deep learning
- Treinamento de modelos usando dados históricos de séries temporais
- Previsões baseadas em modelos treinados
- Integração com componentes FIWARE como Orion Context Broker e STH-Comet

## 🏗 Arquitetura

O projeto segue os princípios da **Arquitetura Limpa (Clean Architecture)** e é organizado em camadas bem definidas:

### Estrutura de Camadas

- **Domain Layer (Camada de Domínio)**
  Contém as entidades centrais e regras de negócio, independente de frameworks externos.
  - `src/domain/entities/`: Definições das entidades como Model, ModelType, ModelStatus
  - `src/domain/repositories/`: Interfaces para repositórios (portas)
  - `src/domain/errors/`: Exceções específicas do domínio

- **Application Layer (Camada de Aplicação)**
  Orquestra o fluxo de dados entre a camada de domínio e o exterior, implementando casos de uso.
  - `src/application/dtos/`: Objetos de transferência de dados
  - `src/application/use_cases/`: Implementação dos casos de uso

- **Infrastructure Layer (Camada de Infraestrutura)**
  Implementa interfaces definidas na camada de domínio com tecnologias específicas.
  - `src/infrastructure/database/`: Implementações de acesso à base de dados
  - `src/infrastructure/repositories/`: Implementações concretas de repositórios

- **Presentation Layer (Camada de Apresentação)**
  Lida com a interação do usuário e apresentação dos dados.
  - `src/presentation/controllers/`: Endpoints da API REST

- **Main Layer (Camada Principal)**
  Configura e inicializa a aplicação, conectando todas as camadas.
  - `src/main/`: Configuração e inicialização da aplicação

### Estrutura de Diretórios

```
fiware-chronos/
├── deploy/                # Arquivos de implantação (Docker, etc.)
├── docs/                  # Documentação
├── scripts/               # Scripts utilitários
├── src/                   # Código fonte principal
│   ├── domain/            # Camada de domínio
│   ├── application/       # Camada de aplicação
│   ├── infrastructure/    # Camada de infraestrutura
│   ├── presentation/      # Camada de apresentação
│   └── main/              # Configuração e inicialização
└── tests/                 # Testes
```

## 🔧 Tecnologias Utilizadas

### Backend

- **FastAPI** (v0.116.1): Framework web assíncrono de alta performance para construção de APIs
- **Pydantic** (v2.11.7): Validação de dados e configurações
- **MongoDB** (v4.14): Banco de dados NoSQL para armazenamento de modelos e dados
- **Celery** (v5.5.3): Sistema de filas para processamento assíncrono (treinamento de modelos)
- **Redis** (v6.4.0): Cache e broker para Celery
- **RabbitMQ**: Message broker para comunicação entre serviços
- **Dependency Injector** (v4.48.1): Container de injeção de dependências

### Ferramentas de Desenvolvimento

- **Python** (v3.12): Linguagem de programação principal
- **Black** (v25.1.0): Formatador de código
- **isort** (v6.0.1): Organizador de imports
- **flake8** (v7.3.0): Linter de código
- **mypy** (v1.17.1): Verificação estática de tipos
- **pre-commit** (v4.3.0): Hooks de pré-commit para garantir qualidade do código

### Integração FIWARE

- **Orion Context Broker**: Gerenciamento de contexto
- **STH-Comet**: Armazenamento de histórico de séries temporais

## 💻 Instalação e Configuração

### Configuração do WSL (para usuários Windows)

Se estiver usando Windows, recomenda-se configurar o WSL (Windows Subsystem for Linux):

```bash
# Configurar WSL versão 2
wsl --set-default-version 2
wsl --install -d Ubuntu
wsl --set-default Ubuntu
```

### Configuração do Ambiente de Desenvolvimento

```bash
# Atualizar o sistema
sudo apt update && sudo apt -y upgrade

# Instalar ferramentas necessárias
sudo apt -y install git make build-essential python3 python3-venv python3-pip

# Clonar e acessar o repositório
git clone https://github.com/tcc-chronos/fiware-chronos.git
cd fiware-chronos

# Criar ambiente virtual Python
python3 -m venv .venv
source .venv/bin/activate

# Instalar as dependências
pip install -r requirements.txt

# Habilitar os scripts
chmod +x scripts/*.sh

# Configurar hooks de pre-commit
pre-commit install
pre-commit run --all-files
```

### Configuração do Docker

Para executar o projeto com todas as suas dependências, utilize o Docker Compose:

1. Certifique-se de ter o Docker e o Docker Compose instalados
2. Copie o arquivo `.env.example` para `.env` e ajuste as configurações conforme necessário
3. Execute o ambiente usando o comando:
   ```bash
   make up ARGS="--build -d"
   ```

## 📡 API Endpoints

### Modelos

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/models` | Listar todos os modelos (paginado) |
| GET | `/models/{model_id}` | Obter detalhes de um modelo específico |
| POST | `/models` | Criar um novo modelo |
| PATCH | `/models/{model_id}` | Atualizar um modelo existente |
| DELETE | `/models/{model_id}` | Excluir um modelo |

### Parâmetros de Consulta

- **GET `/models`**:
  - `skip` (int, opcional): Número de registros a pular (padrão: 0)
  - `limit` (int, opcional): Número máximo de registros a retornar (padrão: 100)
  - `model_type` (string, opcional): Filtrar por tipo de modelo (ex: 'lstm', 'gru')
  - `model_status` (string, opcional): Filtrar por status do modelo (ex: 'draft', 'trained')
  - `entity_id` (string, opcional): Filtrar por ID da entidade FIWARE
  - `feature` (string, opcional): Filtrar por nome do atributo/característica

### Exemplos de Uso

#### Listar modelos com filtros

```bash
# Listar todos os modelos do tipo 'lstm' que estão treinados
curl -X GET "http://localhost:8000/models?model_type=lstm&model_status=trained" \
  -H "accept: application/json"

# Buscar modelos para uma entidade específica e um atributo específico
curl -X GET "http://localhost:8000/models?entity_id=urn:ngsi-ld:Device:001&feature=temperature" \
  -H "accept: application/json"
```

#### Criar um novo modelo

```bash
curl -X POST "http://localhost:8000/models" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Modelo de Previsão de Temperatura",
    "description": "LSTM para previsão de temperatura",
    "model_type": "lstm",
    "dense_dropout": 0.2,
    "rnn_dropout": 0,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "lookback_window": 24,
    "forecast_horizon": 6,
    "feature": "temperature",
    "rnn_units": [128, 64],
    "dense_units": [64, 32],
    "entity_type": "Sensor",
    "entity_id": "urn:ngsi-ld:Chronos:ESP32:001"
    // early_stopping_patience is calculated automatically if not provided
  }'
```

#### Atualizar um modelo existente

```bash
curl -X PATCH "http://localhost:8000/models/3fa85f64-5717-4562-b3fc-2c963f66afa6" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Modelo Atualizado de Temperatura",
    "description": "Modelo atualizado para previsão de temperatura",
    "rnn_units": [128, 64],
    "dense_units": [64, 32],
    "epochs": 150,
    "batch_size": 64,
    "rnn_dropout": 0.3,
    "learning_rate": 0.0005,
    "feature": "temperatura"
  }'
```

## 📊 Modelos de Dados

### ModelType (Enum)

- `LSTM`: Modelo Long Short-Term Memory
- `GRU`: Modelo Gated Recurrent Unit
- `CNN_LSTM`: Modelo híbrido CNN-LSTM

### ModelStatus (Enum)

- `DRAFT`: Modelo criado mas não treinado
- `TRAINING`: Modelo em processo de treinamento
- `TRAINED`: Modelo treinado com sucesso
- `ERROR`: Erro durante treinamento ou uso do modelo

### Model

Principais atributos:
- `id`: Identificador único do modelo
- `name`: Nome do modelo (opcional, será gerado se não fornecido)
- `description`: Descrição do modelo (opcional, será gerado se não fornecido)
- `model_type`: Tipo de arquitetura do modelo (LSTM, GRU, etc.)
- `status`: Status atual do modelo

Hiperparâmetros:
- `rnn_dropout`: Taxa de dropout para conexões recorrentes nas camadas RNN
- `dense_dropout`: Taxa de dropout para camadas densas
- `batch_size`: Tamanho do batch para treinamento
- `epochs`: Número de épocas de treinamento
- `learning_rate`: Taxa de aprendizado
- `validation_split`: Proporção de dados para validação
- `rnn_units`: Lista de unidades para cada camada RNN (obrigatório ter pelo menos um valor positivo)
- `dense_units`: Lista de unidades para cada camada densa
- `bidirectional`: Se deve usar camadas RNN bidirecionais
- `early_stopping_patience`: Número de épocas sem melhoria para parar treinamento (calculado automaticamente se não fornecido)

Configuração de entrada/saída:
- `lookback_window`: Tamanho da janela de histórico
- `forecast_horizon`: Horizonte de previsão
- `feature`: Nome do atributo a ser previsto (do STH Comet)

Configuração do FIWARE:
- `entity_type`: Tipo da entidade no FIWARE
- `entity_id`: ID da entidade no FIWARE

Outros:
- `metadata`: Metadados adicionais
- `trainings`: Histórico de treinamentos do modelo

## 🛠 Desenvolvimento

### Adicionando Novos Componentes

Para adicionar novos componentes, siga a estrutura de camadas da Clean Architecture:

1. Defina as entidades na camada de domínio
2. Crie interfaces (portas) para os novos componentes
3. Implemente os casos de uso na camada de aplicação
4. Adicione implementações concretas na camada de infraestrutura
5. Exponha os recursos na camada de apresentação
6. Conecte tudo na camada principal usando injeção de dependências

### Executando Testes

```bash
# Executar testes unitários
python -m pytest tests/unit

# Executar testes de integração
python -m pytest tests/integration

# Executar testes e2e
python -m pytest tests/e2e
```

## 📝 Comandos Úteis

```bash
# Iniciar os serviços em containers
make up ARGS="--build -d"   # sobe docker compose com os parametros opcionais

# Parar os containers
make stop                   # interrompe os containers

# Executar o servidor local com uvicorn
make run                    # roda uvicorn local com env

# Formatar o código (black + isort)
make format                 # black + isort

# Verificar o código (flake8 + mypy)
make lint                   # flake8 + mypy
```

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
