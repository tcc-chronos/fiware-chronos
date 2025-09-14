# Configuração de Logging

O FIWARE Chronos utiliza um sistema de logging flexível que pode ser configurado via variáveis de ambiente. O sistema suporta diferentes formatos de saída dependendo do ambiente de execução.

## Configuração via Variáveis de Ambiente

| Variável | Descrição | Valor Padrão |
|----------|-----------|--------------|
| `LOG_LEVEL` | Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `LOG_FORMAT` | Formato de log para ambientes não-produção | %(asctime)s - %(name)s - %(levelname)s - %(message)s |
| `LOG_FILE_PATH` | Caminho para arquivo de log (opcional) | None (log apenas para console) |

## Formatos de Log

- **Ambiente de Desenvolvimento**: Logs formatados em texto simples e legível
- **Ambiente de Produção**: Logs formatados em JSON para facilitar integração com ferramentas de análise

## Exemplo de Uso

```python
from src.shared import get_logger

# Obter um logger estruturado
logger = get_logger(__name__)

# Logs simples
logger.info("Operação iniciada")
logger.debug("Valor da variável: x=10")

# Logs com contexto adicional
logger.info("Requisição recebida", method="GET", path="/api/models")
logger.error("Falha na operação", error_code=500, user_id="12345")

# Logger com contexto vinculado
user_logger = logger.bind(user_id="12345", session_id="abc-123")
user_logger.info("Usuário realizou login")
```

## Saída de Logs

### Ambiente de Desenvolvimento
```
2023-09-14 10:15:32,123 - myapp.module - INFO - Operação iniciada
2023-09-14 10:15:32,125 - myapp.module - INFO - Requisição recebida
```

### Ambiente de Produção
```json
{"timestamp": "2023-09-14T10:15:32.123Z", "level": "INFO", "logger": "myapp.module", "message": "Operação iniciada", "app": "fiware-chronos"}
{"timestamp": "2023-09-14T10:15:32.125Z", "level": "INFO", "logger": "myapp.module", "message": "Requisição recebida", "app": "fiware-chronos", "method": "GET", "path": "/api/models"}
```
