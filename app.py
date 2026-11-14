"""Interfaz web (Gradio) del ampliador de fotos familiares.

Ejecutar en local:   uv run app.py
Ejecutar en Colab:   python app.py --share   (genera un link público temporal)

Flujo: cargar foto -> preparar -> bicúbica (clásico) -> difusión (IA)
       -> mapa de lo inventado -> análisis con CLIP -> informe + descarga.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import gradio as gr

from restaurador import analisis, clasico, difusion, textos

CARPETA_EJEMPLOS = Path(__file__).parent / "ejemplos"


def procesar(imagen, nivel, semilla, progress=gr.Progress()):
    if imagen is None:
        raise gr.Error("Primero subí una foto (o elegí uno de los ejemplos de abajo).")
    if nivel not in textos.NIVELES:
        raise gr.Error("Elegí cuánto puede inventar la IA.")

    progress(0.05, desc="Preparando tu foto…")
    lado = difusion.lado_maximo()
    original, fue_reducida = clasico.preparar_imagen(imagen, lado)

    progress(0.10, desc="Ampliando con el método clásico (solo tus píxeles)…")
    bicubica = clasico.ampliar_bicubica(original)

    progress(0.15, desc="Cargando el modelo de IA (la primera vez descarga ~3 GB y tarda varios minutos)…")
    difusion.cargar_pipeline()

    progress(0.30, desc="La IA está reconstruyendo detalles, quitando ruido paso a paso…")
    try:
        con_ia = difusion.ampliar_con_ia(original, textos.NIVELES[nivel], semilla=int(semilla))
    except RuntimeError as error:  # incluye "out of memory" de la GPU
        if "out of memory" in str(error).lower():
            raise gr.Error("Se terminó la memoria de la placa de video. Probá con una foto más chica.") from error
        raise

    progress(0.75, desc="Comparando: ¿qué agregó la IA que no estaba en tu foto?")
    dif = clasico.diferencia(bicubica, con_ia)
    mapa = clasico.colorear_mapa(dif, con_ia)
    pct = clasico.porcentaje_cambiado(dif)
    fidelidad = clasico.fidelidad_al_original(original, con_ia)

    progress(0.85, desc="Revisando el contenido con un segundo modelo (CLIP)…")
    comparacion = analisis.comparar(original, con_ia)

    progress(0.95, desc="Preparando el archivo para descargar…")
    archivo = clasico.exportar_con_aviso(con_ia, {
        "Software": f"restaurador-tpi2 + {difusion.MODELO_DIFUSION}",
        "Nivel": nivel,
        "Semilla": semilla,
    })
    informe = textos.informe(
        nivel=nivel, semilla=int(semilla), fue_reducida=fue_reducida, lado=lado,
        pct_cambiado=pct, fidelidad=fidelidad, comparacion=comparacion,
        hay_gpu=difusion.hay_gpu(),
    )
    return (bicubica, con_ia), mapa, informe, comparacion["tabla"], str(archivo)


def otra_version(imagen, nivel, _semilla_actual, progress=gr.Progress()):
    """Misma foto, mismo nivel, otro ruido inicial: muestra qué cambia (= qué es inventado)."""
    nueva = random.randint(1, 9999)
    return (nueva, *procesar(imagen, nivel, nueva, progress))


def crear_interfaz() -> gr.Blocks:
    with gr.Blocks(title="Ampliador de fotos familiares") as demo:
        gr.Markdown(textos.INTRO)
        gr.Markdown(textos.AVISO_FIJO)

        with gr.Row():
            with gr.Column(scale=1):
                entrada = gr.Image(label="1. Tu foto", type="pil", sources=["upload", "clipboard"])
                nivel = gr.Radio(
                    choices=list(textos.NIVELES),
                    value=textos.NIVEL_POR_DEFECTO,
                    label="2. ¿Cuánto puede inventar la IA?",
                    info="Más invención = foto más nítida, pero menos parecida a la real.",
                )
                with gr.Accordion("Opciones avanzadas", open=False):
                    semilla = gr.Number(
                        value=1234, precision=0, label="Número de versión (semilla)",
                        info="Con el mismo número, la misma foto da siempre el mismo resultado.",
                    )
                boton = gr.Button("3. Ampliar foto", variant="primary")
                boton_otra = gr.Button("Generar otra versión (para ver qué cambia)")

            with gr.Column(scale=2):
                comparador = gr.ImageSlider(
                    label="Antes / después · izquierda: método clásico · derecha: con IA (deslizá)",
                    type="pil",
                )
                mapa = gr.Image(label="Mapa: en rojo, lo que dibujó la IA (no estaba en tu foto)",
                                type="pil")

        informe = gr.Markdown()
        tabla = gr.Dataframe(
            headers=["Grupo", "La foto parece…", "Antes (%)", "Después (%)", "Cambio"],
            label="Análisis de CLIP: original vs. ampliada con IA",
            interactive=False,
        )
        descarga = gr.File(label="Descargar foto ampliada (incluye el aviso «Ampliada con IA»)")

        salidas = [comparador, mapa, informe, tabla, descarga]
        boton.click(procesar, inputs=[entrada, nivel, semilla], outputs=salidas)
        boton_otra.click(otra_version, inputs=[entrada, nivel, semilla], outputs=[semilla, *salidas])

        ejemplos = sorted(str(p) for p in CARPETA_EJEMPLOS.glob("*.jpg"))
        if ejemplos:
            gr.Examples(examples=ejemplos, inputs=entrada, label="Fotos de ejemplo")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ampliador de fotos familiares con IA")
    parser.add_argument("--share", action="store_true", help="link público temporal (para Colab)")
    args = parser.parse_args()
    crear_interfaz().launch(share=args.share)
