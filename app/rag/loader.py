"""Carga los CSV del negocio y los convierte en documentos de texto indexables.

La clave del retrieval está acá: un embedding de `{"Producto": "Malbec", "Precio": 1200}`
serializado como dict matchea mucho peor con "¿cuánto sale el malbec?" que una frase en
lenguaje natural. Por eso cada fila se redacta como una oración con sus campos nombrados.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class DataLoadError(RuntimeError):
    """Los datos del negocio no se pudieron cargar."""


@dataclass(frozen=True)
class Document:
    """Una unidad indexable: el texto que se embebe más su procedencia."""

    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


def _read_csv(path: Path, expected_columns: set[str]) -> pd.DataFrame:
    """Lee un CSV validando que exista, tenga filas y traiga las columnas esperadas."""
    if not path.exists():
        raise DataLoadError(
            f"No se encontró el archivo de datos '{path.name}' en {path.parent}. "
            "Verificá que el directorio `data/` esté completo."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"El archivo '{path.name}' está vacío.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"El archivo '{path.name}' está mal formado: {exc}") from exc

    faltantes = expected_columns - set(df.columns)
    if faltantes:
        raise DataLoadError(
            f"Al archivo '{path.name}' le faltan las columnas: {', '.join(sorted(faltantes))}."
        )
    if df.empty:
        raise DataLoadError(f"El archivo '{path.name}' no tiene filas de datos.")
    return df


def cargar_precios(path: Path) -> list[Document]:
    """Convierte la lista de precios en documentos.

    Cada producto se redacta con su categoría y precio para que consultas como
    "vinos tintos baratos" o "cuánto sale el malbec" encuentren la fila correcta.
    """
    df = _read_csv(path, {"Categoría", "Producto", "Precio"})
    documentos = []
    for _, row in df.iterrows():
        producto = str(row["Producto"]).strip()
        categoria = str(row["Categoría"]).strip()
        precio = str(row["Precio"]).strip()
        texto = f"Producto: {producto}. Categoría: {categoria}. Precio: ${precio} pesos por unidad."
        documentos.append(
            Document(
                text=texto,
                source="precios",
                metadata={"producto": producto, "categoria": categoria, "precio": precio},
            )
        )
    return documentos


_RANGO_HORARIO = re.compile(r"(\d{1,2}:\d{2})\s*a\s*(\d{1,2}:\d{2})")


def _redactar_horarios(horarios: str) -> str:
    """Convierte `"Lunes a Sábado: 09:00 a 19:00, Domingo: Cerrado"` en una frase.

    El formato del CSV deja implícito cuál de los dos números es la apertura, y
    llama3.2:3b se confundía: a "¿a qué hora abren los sábados?" contestaba
    "abren hasta las 19:00", que es el cierre. Nombrando cada extremo, el dato
    deja de depender de que el modelo infiera el orden.

    Si el formato no matchea, se devuelve el texto original: es preferible un
    horario menos redactado que uno mal interpretado.
    """
    rango = _RANGO_HORARIO.search(horarios)
    if not rango:
        return f"Horarios de atención: {horarios}."
    apertura, cierre = rango.groups()
    dias = horarios.split(":")[0].strip()
    frase = f"Abre de {dias.lower()} a las {apertura} y cierra a las {cierre}."
    if "cerrado" in horarios.lower():
        frase += " Los domingos permanece cerrada."
    return frase


def cargar_ubicaciones(path: Path) -> list[Document]:
    """Convierte el listado de sucursales en documentos con dirección y horarios."""
    df = _read_csv(path, {"Ciudad", "Sucursal", "Dirección", "Horarios"})
    documentos = []
    for _, row in df.iterrows():
        sucursal = str(row["Sucursal"]).strip()
        ciudad = str(row["Ciudad"]).strip()
        direccion = str(row["Dirección"]).strip()
        horarios = str(row["Horarios"]).strip()
        texto = (
            f"Sucursal: {sucursal}, en la ciudad de {ciudad}. "
            f"Dirección: {direccion}. "
            f"{_redactar_horarios(horarios)}"
        )
        documentos.append(
            Document(
                text=texto,
                source="ubicaciones",
                metadata={
                    "sucursal": sucursal,
                    "ciudad": ciudad,
                    "direccion": direccion,
                    "horarios": horarios,
                },
            )
        )
    return documentos


def cargar_documentos(data_dir: Path) -> list[Document]:
    """Carga todos los documentos del negocio desde `data_dir`.

    Raises:
        DataLoadError: si falta un archivo, está vacío o le faltan columnas.
    """
    return [
        *cargar_precios(data_dir / "precios.csv"),
        *cargar_ubicaciones(data_dir / "ubicaciones.csv"),
    ]
