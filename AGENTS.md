# Instrucciones para agentes de código

Proyecto: TPI 2 de Procesamiento Digital de Imágenes (IFTS 24). Ampliador x4 de fotos
familiares con difusión que muestra qué inventó la IA. Trabajo individual: la persona
autora tiene que poder explicar cada línea.

## Stack
- Python 3.12 con `uv` (no usar pip ni venv a mano en local).
- Gradio 6 (interfaz), diffusers (`stabilityai/stable-diffusion-x4-upscaler`),
  transformers (`openai/clip-vit-base-patch32`), Pillow y NumPy.
- La difusión corre en Google Colab (GPU T4). La PC local tiene 2 GB de VRAM.

## Comandos
- Instalar: `uv sync`
- Pruebas: `uv run pytest`
- App local: `uv run app.py` · en Colab: `python app.py --share`

## Estructura
- `app.py`: solo interfaz y orquestación del flujo.
- `restaurador/clasico.py`: PDI clásico (bicúbica, diferencias, PSNR, exportar). Sin IA.
- `restaurador/difusion.py`: modelo de difusión. `restaurador/analisis.py`: CLIP.
- `restaurador/textos.py`: todo texto que ve la persona usuaria.

## Reglas
- Todo en español, con nombres de variables en español y comentarios que expliquen el porqué.
- Textos para la persona usuaria (Norma): sin jerga técnica, sin prometer fidelidad.
- Nunca presentar el resultado de la IA como "recuperado": siempre "reconstruido/inventado".
- Cada función clásica nueva lleva su prueba en `tests/` (sin descargar modelos).
- Cambios chicos, un commit por cambio, y mostrar el plan antes de escribir código.
- No subir modelos, imágenes generadas ni `.venv` al repositorio.
