# Clean Architecture - Separação de Responsabilidades no MongoDB

Este documento explica a abordagem adotada para separar as responsabilidades entre o repositório e o acesso direto à base de dados, seguindo os princípios da Clean Architecture.

## Estrutura Implementada

### 1. MongoDatabase (src/infrastructure/database/mongo_database.py)

Esta classe é responsável pela conexão e operações de baixo nível com o MongoDB:

- Gerenciamento da conexão com o MongoDB
- Acesso direto às coleções
- Operações CRUD básicas de baixo nível
- Criação de índices
- Não conhece entidades de domínio, apenas trabalha com dicionários (documentos)

### 2. MongoDBModelRepository (src/infrastructure/repositories/mongodb_model_repository.py)

Esta classe implementa a interface `ModelRepository` e é responsável por:

- Usar o `MongoDatabase` para acessar o MongoDB
- Converter entre entidades de domínio e documentos MongoDB
- Implementar a lógica específica de acesso aos dados para as entidades de modelo
- Lidar com erros específicos de domínio (ModelNotFoundError, ModelOperationError)
- Definir índices específicos para o modelo

## Benefícios da Separação

1. **Seguindo princípios SOLID**:
   - Princípio de Responsabilidade Única (SRP): Cada classe tem uma única responsabilidade
   - Princípio de Inversão de Dependência (DIP): O repositório depende de abstrações, não de implementações concretas

2. **Melhor Testabilidade**:
   - MongoDatabase pode ser mockado facilmente para testes
   - MongoDBModelRepository pode ser testado de forma isolada

3. **Reutilização**:
   - A classe MongoDatabase pode ser reutilizada por outros repositórios
   - Funcionalidades comuns de acesso ao MongoDB estão centralizadas

4. **Manutenibilidade**:
   - Mudanças na lógica de acesso ao banco afetam apenas MongoDatabase
   - Mudanças na lógica de negócios afetam apenas o repositório

5. **Separação clara entre infraestrutura e domínio**:
   - MongoDatabase é puramente infraestrutura
   - MongoDBModelRepository conecta infraestrutura com domínio

## Configuração no Container de DI

```python
# Configuração no container.py
mongo_database = providers.Singleton(
    MongoDatabase,
    mongo_uri=config.database.mongo_uri,
    db_name=config.database.database_name,
)

model_repository = providers.Singleton(
    MongoDBModelRepository,
    mongo_database=mongo_database,
)
```

## Extensibilidade

Esta estrutura permite adicionar novos repositórios de forma simples:

1. Crie uma nova interface de repositório na camada de domínio
2. Implemente a interface usando o MongoDatabase
3. Configure o novo repositório no container de DI

Isso mantém a separação de responsabilidades e facilita a manutenção do código ao longo do tempo.
