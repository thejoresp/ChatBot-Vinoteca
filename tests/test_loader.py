"""Tests de la carga de datos del negocio."""

import pytest

from app.rag.loader import DataLoadError, cargar_documentos, cargar_precios, cargar_ubicaciones


def test_carga_todos_los_documentos(data_dir):
    documentos = cargar_documentos(data_dir)
    assert len(documentos) == 6  # 4 productos + 2 sucursales
    assert {d.source for d in documentos} == {"precios", "ubicaciones"}


def test_producto_se_redacta_en_lenguaje_natural(data_dir):
    """El texto indexado tiene que ser una oración, no un dict serializado.

    Es lo que hace que el embedding matchee preguntas como "cuánto sale el malbec".
    """
    documentos = cargar_precios(data_dir / "precios.csv")
    malbec = next(d for d in documentos if "Malbec" in d.text)

    assert malbec.text == (
        "Producto: Malbec (Bodega Norton). Categoría: Vino Tinto. Precio: $1200 pesos por unidad."
    )
    assert malbec.metadata["precio"] == "1200"
    assert malbec.metadata["categoria"] == "Vino Tinto"


def test_sucursal_incluye_direccion_y_horarios(data_dir):
    documentos = cargar_ubicaciones(data_dir / "ubicaciones.csv")
    cordoba = next(d for d in documentos if "Córdoba" in d.text)

    assert "Av. Colón 789" in cordoba.text
    assert cordoba.metadata["ciudad"] == "Córdoba"
    # El horario original queda intacto en la metadata, aunque el texto se redacte.
    assert cordoba.metadata["horarios"] == "Lunes a Sábado: 09:00 a 19:00, Domingo: Cerrado"


def test_horarios_nombran_apertura_y_cierre(data_dir):
    """Regresión: con "09:00 a 19:00" el modelo respondía "abren hasta las 19:00".

    Nombrar cada extremo saca del medio la inferencia sobre cuál número es cuál.
    """
    documentos = cargar_ubicaciones(data_dir / "ubicaciones.csv")
    cordoba = next(d for d in documentos if "Córdoba" in d.text)

    assert "abre de lunes a sábado a las 09:00" in cordoba.text.lower()
    assert "cierra a las 19:00" in cordoba.text.lower()
    assert "domingos permanece cerrada" in cordoba.text.lower()


def test_horario_con_formato_inesperado_se_deja_como_esta(tmp_path):
    """Ante un formato que no se entiende, no se adivina: se pasa tal cual."""
    (tmp_path / "ubicaciones.csv").write_text(
        "Ciudad,Sucursal,Dirección,Horarios\n"
        'Salta,Sucursal Norte,"Calle 1","Consultar por teléfono"\n',
        encoding="utf-8",
    )
    documento = cargar_ubicaciones(tmp_path / "ubicaciones.csv")[0]

    assert "Consultar por teléfono" in documento.text


def test_archivo_faltante_da_error_accionable(tmp_path):
    with pytest.raises(DataLoadError, match="No se encontró el archivo"):
        cargar_documentos(tmp_path)


def test_archivo_vacio_da_error(tmp_path):
    (tmp_path / "precios.csv").write_text("", encoding="utf-8")
    with pytest.raises(DataLoadError, match="vacío"):
        cargar_precios(tmp_path / "precios.csv")


def test_columnas_faltantes_dan_error(tmp_path):
    (tmp_path / "precios.csv").write_text("Producto,Precio\nMalbec,1200\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="Categoría"):
        cargar_precios(tmp_path / "precios.csv")


def test_csv_sin_filas_da_error(tmp_path):
    (tmp_path / "precios.csv").write_text("Categoría,Producto,Precio\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="no tiene filas"):
        cargar_precios(tmp_path / "precios.csv")
