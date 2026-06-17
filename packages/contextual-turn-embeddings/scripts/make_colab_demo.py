#!/usr/bin/env python3
"""Genera el notebook DEMO para Google Colab (pedagógico, con emojis).

Carga f1 (Dialog2Flow) + f2 (nuestro encoder contextual, desde Hugging Face) y muestra:
  1) disambiguación contextual de un turno, 2) memoria/retrieval, 3) heatmap de atención turno-a-turno.

No se ejecuta acá (tiene celdas propias de Colab: pip install / HF). Se corre en Colab.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)
REPO = "github.com/jumafernandez/doctorado-unsl"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1nCzdwDlZgB_fffUYBnL-DDf15YYKeJoD"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


cells = [
    md(r"""
# 🗣️ Embeddings contextuales de turno — demo en Colab

¿Qué significa un turno de diálogo? **Depende del contexto.** El mismo `"sí"` no es lo mismo si
confirma una reserva o si acepta una cancelación. 🤔

Igual que el ejemplo del *"banco"* (que cambia de sentido según la frase), pero a nivel **turno**:

- 🧩 **f1 (base):** `dialog2flow-joint-bert-base` → `e_t` = embedding del turno **sin contexto**.
- 🧠 **f2 (nuestro):** `contextual-turn-encoder-base` → `h_t` = embedding **contextual** (un Transformer
  *sobre turnos*, no sobre palabras).

En este notebook vamos a ver **3 cositas**: 🎯 disambiguación · 🧠 memoria · 🔥 mapa de atención.
"""),
    md("## ⚙️ 1. Instalación\n\nInstalamos el encoder base, el Hub de Hugging Face y **nuestro paquete** (desde GitHub)."),
    code(r"""
!pip install -q sentence-transformers gdown
!pip install -q "git+https://%s#subdirectory=packages/contextual-turn-embeddings"
""" % REPO),
    md(r"""
## 🤖 2. Cargar los modelos

- **f1** (base) se baja solo de Hugging Face.
- **f2** (nuestro) lo bajamos de una **carpeta pública de Google Drive** con `gdown` (sin login). 📁

> 💡 Alternativa: si preferís, podés montar tu Drive
> (`from google.colab import drive; drive.mount('/content/drive')`) y apuntar a la carpeta directo,
> sin descargar.
"""),
    code(r"""
import os, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, torch, gdown
from sentence_transformers import SentenceTransformer
from contextual_turn_embeddings import ContextualTurnModel

# 📁 carpeta pública de Drive con la estructura models/ (incluye los best/)
DRIVE_FOLDER = "%s"
gdown.download_folder(DRIVE_FOLDER, output="modelos_drive", quiet=False, use_cookies=False)

def best(variante):
    # localiza el checkpoint best/ sin importar cómo quedó anidada la carpeta
    hits = glob.glob(f"modelos_drive/**/contextual-turn-encoder-base-{variante}/best", recursive=True)
    assert hits, f"no encontré el best/ de '{variante}' en la carpeta de Drive"
    return hits[0]

f1 = SentenceTransformer("sergioburdisso/dialog2flow-joint-bert-base")   # base e_t
f2_ar   = ContextualTurnModel.from_pretrained(best("ar-full"))           # causal
f2_bidi = ContextualTurnModel.from_pretrained(best("bidi-full"))         # full-context
print("✅ modelos cargados")
""" % DRIVE_FOLDER_URL),
    md("### 🔧 Helpers\n\nUna función para codificar un diálogo (lista de turnos `(speaker, texto)`) y obtener `e_t` y `h_t`."),
    code(r"""
import pandas as pd
from contextual_turn_embeddings import encode_dialogues

def encode_dialogo(turnos, modelo):
    # turnos = [(speaker, texto), ...] -> (e_t, h_t) por turno
    df = pd.DataFrame({"dialogue_id": "d", "turn_id": range(len(turnos)),
                       "speaker": [s for s, _ in turnos],
                       "utterance": [u for _, u in turnos]})
    e = np.asarray(f1.encode([u for _, u in turnos]), dtype=np.float32)
    h, meta = encode_dialogues(modelo, df, embeddings=e, device="cpu")
    return e, h[meta.sort_values("turn_id").index.to_numpy()]

def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)
"""),
    md(r"""
## 🎯 3. El mismo turno, distinto significado

`"Sí, está bien."` 👉 una vez **confirma una reserva**, otra **acepta una cancelación con penalidad**.
El texto es idéntico ⇒ el `e_t` static es el **mismo vector**. ¿Y el contextual? 👀
"""),
    code(r"""
turno = ("user", "Sí, está bien.")
confirma = [("user","Quiero reservar una mesa para 4 a las 20:00."),
            ("system","Tengo mesa para 4 a las 20:00. ¿A nombre de quién?"),
            ("user","A nombre de Juan."),
            ("system","Listo: mesa para 4, 20:00, Juan. ¿Confirmo?"), turno]
cancela = [("user","Necesito cancelar mi vuelo de mañana."),
           ("system","La tarifa tiene 30% de penalidad. ¿Cancela igual?"),
           ("user","¿No hay forma de evitar la penalidad?"),
           ("system","Con esta tarifa no. Procedo con la cancelación y la penalidad."), turno]

e1, h1 = encode_dialogo(confirma, f2_ar)
e2, h2 = encode_dialogo(cancela, f2_ar)
print(f'Turno: "{turno[1]}"\n')
print(f"  cos(e_t  static)     = {cos(e1[-1], e2[-1]):.4f}   🟰 mismo texto => vector IDÉNTICO")
print(f"  cos(h_t  contextual) = {cos(h1[-1], h2[-1]):.4f}   ✅ el modelo lo SEPARA por contexto")
"""),
    md(r"""
**Lectura 📌** El static no distingue (`"sí"` es `"sí"`, cos = 1.0000). El **contextual (AR)** sí, y
fuerte: el mismo turno se **aleja** según confirme o cancele. Eso es lo que aporta el modelo y un
embedding por-turno aislado no puede capturar. 🎉
"""),
    md(r"""
## 🧠 4. Memoria conversacional: ¿qué situación se parece?

Una **consulta** (turno + contexto) y varias **situaciones candidatas**. Recuperamos la más parecida
por coseno — static vs contextual, con scores. 🔎
"""),
    code(r"""
consulta = [("user","Hace una semana mi pedido figura 'en camino' y no llega."),
            ("system","Lamento la demora. ¿Lo reviso?"),
            ("user","Sí, y quiero que me devuelvan la plata.")]
candidatas = {
 "🧾 reclamo / reembolso": [("user","Me cobraron dos veces la suscripción."),
                            ("system","Veo el doble cargo. ¿Cómo procedo?"),
                            ("user","Quiero que me reintegren un cobro.")],
 "👋 saludo inicial":      [("user","Hola, buen día."),("system","¡Hola! ¿En qué te ayudo?"),
                            ("user","Quería hacer una consulta.")],
 "📅 reserva":             [("user","Quiero una mesa para dos el viernes."),
                            ("system","¿A qué horario?"),("user","A las nueve.")],
}
def last(d, contextual): e, h = encode_dialogo(d, f2_ar); return (h if contextual else e)[-1]
qe, qh = last(consulta, False), last(consulta, True)
print("Consulta: reclamo por pedido demorado + pide reembolso\n")
print(f"{'candidata':<24}{'static':>10}{'contextual':>14}")
for n, d in candidatas.items():
    print(f"{n:<24}{cos(qe, last(d, False)):>10.4f}{cos(qh, last(d, True)):>14.4f}")
"""),
    md(r"""
**Lectura honesta 🫡** Recuperación a nivel **situación**. Acá el static suele ordenar bien; el
contextual *zero-shot* todavía **no está calibrado para retrieval de situación** (coincide con la
evaluación MSS@10). La **disambiguación** (🎯) ya funciona; la **similitud-de-situación** es lo que
falta afinar (objetivo contrastivo a nivel situación). Honestidad ante todo. 💪
"""),
    md(r"""
## 🔥 5. ¿A qué turnos "mira" el modelo? (mapa de atención)

Como el heatmap de atención de BERT, pero **entre turnos**. El modelo AR es **causal**: cada turno
solo puede mirar a los **anteriores** (y a sí mismo) → triángulo inferior. 👇
"""),
    code(r"""
import matplotlib.pyplot as plt, seaborn as sns
from contextual_turn_embeddings.utils import build_causal_mask

@torch.no_grad()
def atencion_turnos(modelo, turnos, capa=0):
    e = np.asarray(f1.encode([u for _, u in turnos]), dtype=np.float32)
    spk = torch.tensor([[0 if s == "user" else 1 for s, _ in turnos]])
    emb = torch.tensor(e).unsqueeze(0); S = emb.shape[1]
    x = modelo.input_proj(emb) + modelo.position_embedding(torch.arange(S).unsqueeze(0))
    if modelo.speaker_embedding is not None:
        x = x + modelo.speaker_embedding(spk.clamp(0, modelo.config.num_speakers - 1))
    x = modelo.input_layer_norm(x)
    L = modelo.encoder.layers[capa]; h = L.norm1(x)
    am = build_causal_mask(S, x.device) if modelo.config.attention_mode == "autoregressive" else None
    _, w = L.self_attn(h, h, h, need_weights=True, average_attn_weights=True, attn_mask=am)
    return w[0].numpy()

dialogo = [("user","Hola, quiero reservar una mesa."),("system","¿Para cuántas personas?"),
           ("user","Para cuatro, a las ocho."),("system","¿A nombre de quién?"),("user","Juan.")]
W = atencion_turnos(f2_ar, dialogo)
etiquetas = [f"{s[0]}: {u[:22]}" for s, u in dialogo]

plt.figure(figsize=(7.5, 6))
sns.heatmap(W, xticklabels=etiquetas, yticklabels=etiquetas, cmap="viridis",
            annot=True, fmt=".2f", cbar_kws={"label": "atención"})
plt.title("🔥 Atención turno-a-turno (capa 1) — modelo AR (causal)")
plt.xlabel("turno atendido →"); plt.ylabel("turno que atiende ↓")
plt.tight_layout(); plt.show()
"""),
    md(r"""
**Lectura 📌** Cada fila es un turno y muestra **cuánto mira a los anteriores**. El triángulo superior
está vacío porque el modelo es **causal** (no puede ver el futuro). Así se ve, literal, cómo el turno
actual integra su **historia conversacional**. 🧠✨

---
🔬 *Demo cualitativo.* La evaluación cuantitativa (act-match P@k, MSS@10 con juez LLM, reconciliación
con el paper) vive en `packages/conversational-ann/results/RESULTS.md`.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
path = NB_DIR / "demo_colab.ipynb"
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("escrito", path)
