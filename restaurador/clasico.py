"""Procesamiento digital clásico (sin IA).

Todo lo de este archivo usa SOLO los píxeles de la foto original. Por eso
sirve como "regla" para medir qué agregó el modelo de difusión:

- La interpolación bicúbica agranda la foto estirando los píxeles que ya
  existían: no inventa detalle nuevo, solo lo suaviza.
- Si restamos (IA - bicúbica), lo que queda es aproximadamente lo que la IA
  dibujó por su cuenta.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from PIL.PngImagePlugin import PngInfo

FACTOR = 4  # el modelo x4-upscaler multiplica ancho y alto por 4

# Escala FIJA del mapa de diferencias: una diferencia de 64 niveles (de 255)
# ya se pinta con el rojo más intenso. Usar una escala fija (y no "la máxima
# diferencia de cada foto") permite comparar mapas de fotos distintas.
ESCALA_MAPA = 64.0

# A partir de cuántos niveles de gris consideramos que un píxel "cambió".
UMBRAL_CAMBIO = 25

TEXTO_AVISO = "Ampliada con IA - los detalles finos fueron reconstruidos, no recuperados"


def preparar_imagen(imagen: Image.Image, lado_maximo: int) -> tuple[Image.Image, bool]:
    """Deja la foto lista para procesar.

    1. Gira la foto según la etiqueta EXIF (las fotos de celular suelen
       guardarse "acostadas" con una nota que dice cómo girarlas).
    2. Convierte a RGB (3 canales). Si tiene transparencia, la apoya sobre
       un fondo blanco.
    3. Si algún lado supera ``lado_maximo`` píxeles, la achica manteniendo
       la proporción (el modelo necesita mucha memoria por cada píxel).

    Devuelve la imagen preparada y si hubo que achicarla.
    """
    imagen = ImageOps.exif_transpose(imagen)

    tiene_transparencia = imagen.mode in ("RGBA", "LA") or (
        imagen.mode == "P" and "transparency" in imagen.info
    )
    if tiene_transparencia:
        rgba = imagen.convert("RGBA")
        fondo = Image.new("RGB", rgba.size, (255, 255, 255))
        fondo.paste(rgba, mask=rgba.getchannel("A"))
        imagen = fondo
    else:
        imagen = imagen.convert("RGB")

    fue_reducida = max(imagen.size) > lado_maximo
    if fue_reducida:
        imagen = imagen.copy()
        imagen.thumbnail((lado_maximo, lado_maximo), Image.Resampling.LANCZOS)
    return imagen, fue_reducida


def ampliar_bicubica(imagen: Image.Image, factor: int = FACTOR) -> Image.Image:
    """Línea base clásica: agranda la foto con interpolación bicúbica.

    Cada píxel nuevo se calcula como un promedio ponderado de los 16 píxeles
    vecinos (4x4) de la foto original. Solo usa información que ya estaba.
    """
    nuevo_tamano = (imagen.width * factor, imagen.height * factor)
    return imagen.resize(nuevo_tamano, Image.Resampling.BICUBIC)


def _a_gris(imagen: Image.Image) -> np.ndarray:
    """Pasa a escala de grises (luminancia) como matriz de números decimales."""
    return np.asarray(imagen.convert("L"), dtype=np.float32)


def diferencia(base: Image.Image, comparada: Image.Image) -> np.ndarray:
    """Matriz con la diferencia absoluta de brillo, píxel a píxel (0 a 255).

    Las dos imágenes tienen que tener el mismo tamaño.
    """
    if base.size != comparada.size:
        raise ValueError(f"Tamaños distintos: {base.size} y {comparada.size}")
    return np.abs(_a_gris(comparada) - _a_gris(base))


def colorear_mapa(dif: np.ndarray, fondo: Image.Image) -> Image.Image:
    """Pinta en rojo las zonas donde hubo más diferencia, sobre la foto en gris."""
    intensidad = np.clip(dif / ESCALA_MAPA, 0.0, 1.0)[..., None]  # 0 = igual, 1 = muy distinto
    gris = _a_gris(fondo)[..., None].repeat(3, axis=2) * 0.6  # fondo apagado
    rojo = np.array([255.0, 30.0, 30.0])
    mezcla = gris * (1 - intensidad) + rojo * intensidad
    return Image.fromarray(mezcla.astype(np.uint8))


def porcentaje_cambiado(dif: np.ndarray, umbral: int = UMBRAL_CAMBIO) -> float:
    """Qué porcentaje de la imagen cambió más que ``umbral`` niveles de gris."""
    return float((dif > umbral).mean() * 100)


def fidelidad_al_original(original: Image.Image, ampliada: Image.Image) -> float:
    """Chequeo de consistencia: ¿la foto ampliada, vuelta a achicar, se parece al original?

    Achicamos el resultado al tamaño original y medimos la diferencia promedio
    (en % de la escala 0-255). Un valor bajo significa que la IA respetó lo que
    había en tu foto; un valor alto significa que cambió tonos o formas, no
    solo agregó detalle fino.
    """
    reducida = ampliada.resize(original.size, Image.Resampling.LANCZOS)
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(reducida, dtype=np.float32)
    return float(np.abs(a - b).mean() / 255 * 100)


def psnr(referencia: Image.Image, prueba: Image.Image) -> float:
    """PSNR en decibeles: qué tan parecidos son los píxeles a una referencia real.

    Más alto = más parecido (más de 30 dB suele verse muy similar). Solo se puede
    calcular si tenemos la foto "verdadera" en alta resolución: por eso se usa en
    la prueba de laboratorio (achicar una foto buena, ampliarla y comparar).
    Mide parecido, NO autenticidad.
    """
    a = np.asarray(referencia.convert("RGB"), dtype=np.float64)
    b = np.asarray(prueba.convert("RGB").resize(referencia.size), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse))


def exportar_con_aviso(imagen: Image.Image, detalles: dict[str, str]) -> Path:
    """Guarda un PNG con el aviso visible abajo y el aviso también en los metadatos.

    Así la advertencia viaja con el archivo aunque alguien recorte la franja.
    """
    alto_franja = max(18, imagen.height // 40)
    salida = Image.new("RGB", (imagen.width, imagen.height + alto_franja), (20, 20, 20))
    salida.paste(imagen, (0, 0))
    ImageDraw.Draw(salida).text((6, imagen.height + 3), TEXTO_AVISO, fill=(255, 255, 255))

    metadatos = PngInfo()
    metadatos.add_text("Description", TEXTO_AVISO)
    for clave, valor in detalles.items():
        metadatos.add_text(clave, str(valor))

    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = Path(tempfile.gettempdir()) / f"foto_ampliada_con_IA_{marca}.png"
    salida.save(ruta, pnginfo=metadatos)
    return ruta
