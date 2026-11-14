"""Ampliación x4 con un modelo de difusión de Hugging Face.

Cómo funciona el modelo, en simple:
1. Toma tu foto chica como "guía".
2. Arranca desde una imagen 4 veces más grande llena de ruido.
3. En cada paso le quita un poco de ruido, tratando de que el resultado,
   al achicarlo, se parezca a tu foto.
4. Los detalles finos (pestañas, texturas, letras) NO están en tu foto:
   el modelo los dibuja según lo que aprendió de millones de imágenes.

El parámetro ``noise_level`` controla cuánto "desconfía" el modelo de tu foto:
con valores bajos se apega a ella; con valores altos inventa más.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from restaurador.clasico import FACTOR

MODELO_DIFUSION = "stabilityai/stable-diffusion-x4-upscaler"

# Texto neutro: no le pedimos al modelo ningún estilo ni contenido en particular.
PROMPT = "a photograph"

# Pasos de "limpieza de ruido". Más pasos = más lento; 25 es un buen equilibrio.
PASOS = 25

# Tamaño máximo del lado de la foto de entrada (la salida es 4 veces más grande).
LADO_MAXIMO_GPU = 256  # 256 -> 1024 px, entra cómodo en una GPU T4 de Colab
LADO_MAXIMO_CPU = 128  # sin GPU es MUY lento: limitamos a 128 -> 512 px

_pipeline = None  # el modelo se carga una sola vez y se reutiliza


def hay_gpu() -> bool:
    import torch

    return torch.cuda.is_available()


def lado_maximo() -> int:
    return LADO_MAXIMO_GPU if hay_gpu() else LADO_MAXIMO_CPU


def cargar_pipeline():
    """Descarga (la primera vez, ~3 GB) y carga el modelo en memoria."""
    global _pipeline
    if _pipeline is None:
        import torch
        from diffusers import StableDiffusionUpscalePipeline

        if hay_gpu():
            pipe = StableDiffusionUpscalePipeline.from_pretrained(
                MODELO_DIFUSION, torch_dtype=torch.float16
            ).to("cuda")
        else:
            pipe = StableDiffusionUpscalePipeline.from_pretrained(MODELO_DIFUSION)
        pipe.enable_attention_slicing()  # usa menos memoria a cambio de un poco de velocidad
        pipe.set_progress_bar_config(disable=True)
        _pipeline = pipe
    return _pipeline


def rellenar_a_multiplo(imagen: Image.Image, multiplo: int = 64) -> Image.Image:
    """Agranda el lienzo hasta un múltiplo de ``multiplo`` repitiendo los bordes.

    El modelo trabaja mejor con medidas múltiplos de 64. En lugar de deformar
    la foto, agregamos borde (copiando la última fila/columna) y después lo
    recortamos del resultado.
    """
    ancho, alto = imagen.size
    nuevo_ancho = -(-ancho // multiplo) * multiplo  # redondeo hacia arriba
    nuevo_alto = -(-alto // multiplo) * multiplo
    matriz = np.asarray(imagen)
    relleno = ((0, nuevo_alto - alto), (0, nuevo_ancho - ancho), (0, 0))
    return Image.fromarray(np.pad(matriz, relleno, mode="edge"))


def ampliar_con_ia(imagen: Image.Image, noise_level: int, semilla: int = 1234) -> Image.Image:
    """Amplía la foto x4 con el modelo de difusión.

    ``semilla`` fija el ruido inicial: con la misma semilla, la misma foto y el
    mismo nivel, el resultado se repite. Con otra semilla, cambian los detalles
    inventados (¡y eso demuestra que son inventados!).
    """
    import torch

    pipe = cargar_pipeline()
    ancho, alto = imagen.size
    dispositivo = "cuda" if hay_gpu() else "cpu"
    generador = torch.Generator(device=dispositivo).manual_seed(int(semilla))

    resultado = pipe(
        prompt=PROMPT,
        image=rellenar_a_multiplo(imagen),
        noise_level=int(noise_level),
        num_inference_steps=PASOS,
        generator=generador,
    ).images[0]

    # Quitamos el borde agregado y nos aseguramos del tamaño exacto x4.
    destino = (ancho * FACTOR, alto * FACTOR)
    resultado = resultado.crop((0, 0, *destino))
    if resultado.size != destino:
        resultado = resultado.resize(destino, Image.Resampling.BICUBIC)
    return resultado
