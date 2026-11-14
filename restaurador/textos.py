"""Textos que ve la persona usuaria (Norma): lenguaje simple, sin jerga.

Separar los textos del código permite revisarlos y mejorarlos sin tocar la lógica.
"""

from __future__ import annotations

# Control "¿cuánto puede inventar la IA?" -> valor de noise_level del modelo.
NIVELES = {
    "Poco (más fiel a mi foto)": 10,
    "Medio": 40,
    "Mucho (más nítida, más inventada)": 80,
}
NIVEL_POR_DEFECTO = "Poco (más fiel a mi foto)"

INTRO = """
# 📷 Ampliador de fotos familiares, con honestidad

Subí una foto **chica o escaneada en baja resolución** y la vamos a agrandar 4 veces
de dos maneras:

1. **Método clásico (bicúbica):** estira los puntos que ya tenía tu foto. Se ve más suave, pero **no inventa nada**.
2. **Inteligencia artificial (difusión):** se ve más nítida, pero **los detalles finos los dibuja la IA**.
   No los "recupera": los imagina a partir de millones de fotos que vio antes.

Al final te mostramos **dónde inventó** la IA, para que decidas si la foto sigue siendo *tu* foto.
"""

AVISO_FIJO = (
    "⚠️ **La foto ampliada con IA no es un documento.** Las caras, letras y texturas finas "
    "pueden no ser las reales. Comparala con el original antes de compartirla o archivarla."
)


def informe(*, nivel: str, semilla: int, fue_reducida: bool, lado: int,
            pct_cambiado: float, fidelidad: float, comparacion: dict, hay_gpu: bool) -> str:
    """Arma la explicación en lenguaje simple de lo que pasó con la foto."""
    lineas = ["## ¿Qué hizo la IA con tu foto?", ""]

    lineas.append(
        f"- Tu foto ahora tiene **16 veces más píxeles** (4 veces más ancha y 4 veces más alta). "
        f"Usaste el nivel **«{nivel}»** (versión n.º {semilla})."
    )
    lineas.append(
        f"- En el mapa rojo, **{pct_cambiado:.0f} % de la imagen** tiene detalles que la IA dibujó "
        "por su cuenta (no aparecen en el método clásico)."
    )

    if fidelidad < 3:
        juicio = "respetó muy bien los tonos y las formas de tu foto"
    elif fidelidad < 6:
        juicio = "respetó en general tu foto, con algunos cambios de tono o de forma"
    else:
        juicio = "**cambió bastante** tonos o formas: revisala con cuidado"
    lineas.append(
        f"- Si achicamos la foto ampliada de nuevo, se diferencia un **{fidelidad:.1f} %** del original: "
        f"la IA {juicio}."
    )

    if fue_reducida:
        lineas.append(
            f"- ℹ️ Tu foto era más grande que {lado} px de lado y **la achicamos antes de ampliarla** "
            "(el modelo necesita mucha memoria). Esto ya pierde detalle."
        )
    if not hay_gpu:
        lineas.append("- ℹ️ Esta computadora no tiene placa de video compatible: usamos un tamaño menor.")

    lineas += ["", "### Lo que vio el analizador automático (CLIP)", ""]
    if comparacion["contenido_cambio"]:
        lineas.append(
            f"- 🔴 **Atención:** antes parecía *{comparacion['contenido_antes']}* y después parece "
            f"*{comparacion['contenido_despues']}*. Puede que la IA haya cambiado el contenido."
        )
    else:
        lineas.append(
            f"- ✅ El contenido principal se mantuvo: *{comparacion['contenido_antes']}*."
        )
    if comparacion["hay_rostro"]:
        lineas.append(
            "- 👤 **Hay rostros.** La IA puede cambiar rasgos (ojos, boca, arrugas) y hacer que una persona "
            "parezca más joven o distinta. Mirá las caras en el mapa rojo y compará con el original."
        )
    lineas.append(
        "- *CLIP es un modelo automático que también se equivoca: tomalo como una pista, no como una prueba.*"
    )

    lineas += ["", "### Cómo verificarla",
               "- Compará con otra foto de la misma persona o lugar, o preguntale a alguien que estuvo ahí.",
               "- Probá **«Generar otra versión»**: lo que cambia entre versiones es, sin duda, inventado.",
               "- El archivo descargado lleva el aviso «Ampliada con IA», en la imagen y en sus datos internos."]
    return "\n".join(lineas)
