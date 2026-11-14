"""Análisis visual con CLIP (clasificación zero-shot).

CLIP convierte una imagen y varias frases en vectores (listas de números) y
mide cuánto se parece la imagen a cada frase. "Zero-shot" significa que no lo
entrenamos: solo le damos las frases candidatas.

Lo usamos para COMPARAR la foto original con la ampliada:
- ¿Cambió cómo "se ve" la calidad (nítida, borrosa, pixelada)?
- ¿Cambió el CONTENIDO principal? Si antes era "un rostro" y después
  "un paisaje", algo raro pasó y hay que revisarlo.

Cuidado: CLIP no fue entrenado para medir calidad, y los porcentajes dependen
mucho de las frases elegidas. Es una ayuda, nunca una prueba.
"""

from __future__ import annotations

from PIL import Image

MODELO_CLIP = "openai/clip-vit-base-patch32"

# CLIP aprendió con textos en inglés: le hablamos en inglés y mostramos en español.
ETIQUETAS_CALIDAD = [
    ("nítida y con detalle", "a sharp, detailed photo"),
    ("borrosa", "a blurry photo"),
    ("pixelada", "a pixelated, low resolution photo"),
    ("con ruido o grano", "a noisy, grainy photo"),
]

ETIQUETAS_CONTENIDO = [
    ("rostro de una persona", "a close-up photo of a person's face"),
    ("grupo de personas", "a photo of a group of people"),
    ("paisaje o exterior", "a photo of a landscape or a street"),
    ("objeto o interior", "a photo of an object or a room"),
    ("documento o texto", "a photo of a document with text"),
    ("animal", "a photo of an animal"),
    # Agregada después de una prueba: sin esta opción, CLIP clasificó una
    # locomotora como "animal" (37,7 %). CLIP solo elige entre las frases que le damos.
    ("vehículo o máquina", "a photo of a vehicle, a train or a machine"),
]

ETIQUETA_ROSTRO = "rostro de una persona"
UMBRAL_ROSTRO = 0.30  # si CLIP da más de 30 % a "rostro", avisamos

_clasificador = None


def cargar_clasificador():
    global _clasificador
    if _clasificador is None:
        import torch
        from transformers import pipeline

        _clasificador = pipeline(
            "zero-shot-image-classification",
            model=MODELO_CLIP,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _clasificador


def clasificar(imagen: Image.Image, etiquetas: list[tuple[str, str]]) -> dict[str, float]:
    """Devuelve {etiqueta_en_español: probabilidad} (las probabilidades suman 1)."""
    ingles_a_espanol = {en: es for es, en in etiquetas}
    resultados = cargar_clasificador()(imagen, candidate_labels=list(ingles_a_espanol))
    return {ingles_a_espanol[r["label"]]: float(r["score"]) for r in resultados}


def comparar(antes: Image.Image, despues: Image.Image) -> dict:
    """Clasifica la foto original y la ampliada con las mismas frases."""
    calidad_antes = clasificar(antes, ETIQUETAS_CALIDAD)
    calidad_despues = clasificar(despues, ETIQUETAS_CALIDAD)
    contenido_antes = clasificar(antes, ETIQUETAS_CONTENIDO)
    contenido_despues = clasificar(despues, ETIQUETAS_CONTENIDO)

    principal_antes = max(contenido_antes, key=contenido_antes.get)
    principal_despues = max(contenido_despues, key=contenido_despues.get)

    tabla = []
    for grupo, etiquetas, a, d in (
        ("Calidad", ETIQUETAS_CALIDAD, calidad_antes, calidad_despues),
        ("Contenido", ETIQUETAS_CONTENIDO, contenido_antes, contenido_despues),
    ):
        for es, _ in etiquetas:
            tabla.append([grupo, es, round(a[es] * 100, 1), round(d[es] * 100, 1),
                          round((d[es] - a[es]) * 100, 1)])

    return {
        "tabla": tabla,
        "contenido_antes": principal_antes,
        "contenido_despues": principal_despues,
        "contenido_cambio": principal_antes != principal_despues,
        "hay_rostro": max(contenido_antes[ETIQUETA_ROSTRO],
                          contenido_despues[ETIQUETA_ROSTRO]) >= UMBRAL_ROSTRO,
    }
