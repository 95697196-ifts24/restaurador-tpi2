"""Pruebas de las funciones clásicas (no descargan modelos). Ejecutar: uv run pytest"""

import numpy as np
from PIL import Image

from restaurador import clasico
from restaurador.difusion import SEMILLA_POR_DEFECTO, normalizar_semilla, rellenar_a_multiplo


def foto_de_prueba(ancho=60, alto=40, modo="RGB"):
    rng = np.random.default_rng(0)
    canales = {"RGB": 3, "RGBA": 4}[modo]
    return Image.fromarray(rng.integers(0, 256, (alto, ancho, canales), dtype=np.uint8), modo)


def test_preparar_quita_transparencia():
    imagen, reducida = clasico.preparar_imagen(foto_de_prueba(modo="RGBA"), lado_maximo=256)
    assert imagen.mode == "RGB"
    assert not reducida


def test_preparar_achica_fotos_grandes_manteniendo_proporcion():
    imagen, reducida = clasico.preparar_imagen(foto_de_prueba(600, 300), lado_maximo=256)
    assert reducida
    assert imagen.size == (256, 128)


def test_bicubica_multiplica_por_4():
    assert clasico.ampliar_bicubica(foto_de_prueba(60, 40)).size == (240, 160)


def test_diferencia_es_cero_si_las_imagenes_son_iguales():
    foto = foto_de_prueba()
    dif = clasico.diferencia(foto, foto)
    assert dif.max() == 0
    assert clasico.porcentaje_cambiado(dif) == 0


def test_mapa_tiene_el_mismo_tamano_que_la_foto():
    foto = foto_de_prueba()
    mapa = clasico.colorear_mapa(clasico.diferencia(foto, foto), foto)
    assert mapa.size == foto.size


def test_bicubica_es_fiel_al_original():
    """La bicúbica no inventa: al achicarla de nuevo se parece mucho al original."""
    original = Image.new("RGB", (64, 64), (120, 80, 40))
    assert clasico.fidelidad_al_original(original, clasico.ampliar_bicubica(original)) < 1


def test_psnr_infinito_si_son_iguales_y_baja_con_ruido():
    foto = foto_de_prueba()
    ruido = Image.fromarray(np.clip(np.asarray(foto, dtype=int) + 30, 0, 255).astype(np.uint8))
    assert clasico.psnr(foto, foto) == float("inf")
    assert clasico.psnr(foto, ruido) < 30


def test_exportar_guarda_aviso_en_metadatos():
    ruta = clasico.exportar_con_aviso(foto_de_prueba(), {"Nivel": "Poco"})
    guardada = Image.open(ruta)
    assert guardada.text["Description"] == clasico.TEXTO_AVISO
    assert guardada.text["Nivel"] == "Poco"


def test_relleno_llega_a_multiplo_de_64_sin_tocar_la_foto():
    foto = foto_de_prueba(100, 70)
    rellena = rellenar_a_multiplo(foto)
    assert rellena.size == (128, 128)
    assert np.array_equal(np.asarray(rellena)[:70, :100], np.asarray(foto))


def test_semilla_vacia_usa_la_de_por_defecto():
    """Si se borra el campo "Número de versión", Gradio manda None: no debe romper."""
    assert normalizar_semilla(None) == SEMILLA_POR_DEFECTO
    assert normalizar_semilla(42.0) == 42
