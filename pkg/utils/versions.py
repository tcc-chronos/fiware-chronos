from importlib.metadata import PackageNotFoundError, version


def pkg_version(name: str) -> str:
    """Retorna a versão instalada de um pacote ou 'unknown'."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
