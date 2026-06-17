#!/usr/bin/env python3
"""Genera el notebook demo 'embeddings contextuales de turno en acción'.

Análogo a la demo del 'banco' (BETO), pero a nivel turno de diálogo:
- f1 = dialog2flow-joint-bert-base  -> e_t (static, sin contexto)
- f2 = contextual-turn-encoder-base -> h_t (contextual)

Tras generarlo:
    jupyter nbconvert --to notebook --execute --inplace notebooks/demo_turnos_contextuales.ipynb
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md(r"""
# 🗣️ Embeddings contextuales de turno — en acción

Análogo a la demo del *"banco"* con BETO, pero a nivel **turno de diálogo**: el mismo turno
significa distinto según el **contexto** de la conversación. El embedding **static** (`e_t`,
Dialog2Flow sin contexto) no lo ve; el **contextual** (`h_t`, nuestro modelo) sí.

- **f1 (base):** `dialog2flow-joint-bert-base` → `e_t` (un vector por turno, sin contexto).
- **f2 (nuestro):** `contextual-turn-encoder-base` → `h_t` (Transformer **sobre turnos**, contextual).

> Demo exploratorio/cualitativo. La evaluación cuantitativa está en `packages/conversational-ann/`.
"""),
    code(r"""
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

PKG = Path("~/Documents/GitHub/doctorado-unsl/packages/contextual-turn-embeddings").expanduser()
sys.path.insert(0, str(PKG))
from contextual_turn_embeddings import ContextualTurnModel, encode_dialogues

# f1: encoder base (e_t) — el mismo que las baselines (apples-to-apples)
f1 = SentenceTransformer("sergioburdisso/dialog2flow-joint-bert-base")

# f2: nuestro encoder contextual (h_t). AR = causal; Bidi = full-context.
f2_ar = ContextualTurnModel.from_pretrained(str(PKG / "models/contextual-turn-encoder-base-ar-full/best"))
f2_bidi = ContextualTurnModel.from_pretrained(str(PKG / "models/contextual-turn-encoder-base-bidi-full/best"))
print("modelos cargados ✓")
"""),
    code(r"""
def encode_dialogo(turnos, modelo):
    # turnos = lista de (speaker, utterance). Devuelve (e_t, h_t) por turno, alineados.
    df = pd.DataFrame({"dialogue_id": "d", "turn_id": range(len(turnos)),
                       "speaker": [s for s, _ in turnos],
                       "utterance": [u for _, u in turnos]})
    e = np.asarray(f1.encode([u for _, u in turnos]), dtype=np.float32)
    h, meta = encode_dialogues(modelo, df, embeddings=e, device="cpu")
    order = meta.sort_values("turn_id").index.to_numpy()   # alinear al orden de los turnos
    return e, h[order]

def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)
"""),
    md(r"""
## Demo 1 — El mismo turno, distinto significado según el contexto

`"Sí, está bien."` como **confirmación de una reserva** vs como **aceptación de una cancelación con
penalidad**. El texto es idéntico → el `e_t` static es el mismo vector. ¿Y el contextual?
"""),
    code(r"""
turno_ambiguo = ("user", "Sí, está bien.")

dialogo_confirma = [
    ("user", "Quiero reservar una mesa para 4 personas a las 8 de la noche."),
    ("system", "Tengo una mesa para 4 a las 20:00. ¿A nombre de quién la registro?"),
    ("user", "A nombre de Juan."),
    ("system", "Listo: mesa para 4, 20:00, a nombre de Juan. ¿Confirmo?"),
    turno_ambiguo,
]
dialogo_cancela = [
    ("user", "Necesito cancelar mi vuelo de mañana a Córdoba."),
    ("system", "La tarifa tiene una penalidad de cancelación del 30%. ¿Aun así desea cancelar?"),
    ("user", "¿No hay forma de evitar la penalidad?"),
    ("system", "Con esta tarifa no. Entonces procedo con la cancelación y la penalidad."),
    turno_ambiguo,
]

e1, h1 = encode_dialogo(dialogo_confirma, f2_ar)
e2, h2 = encode_dialogo(dialogo_cancela, f2_ar)

print(f'Turno evaluado: "{turno_ambiguo[1]}"\n')
print(f"  cos(e_t  static)     = {cos(e1[-1], e2[-1]):.4f}   <- mismo texto => vector IDÉNTICO")
print(f"  cos(h_t  contextual) = {cos(h1[-1], h2[-1]):.4f}   <- el modelo lo SEPARA por contexto")
print(f"  divergencia contextual (1 - cos) = {1 - cos(h1[-1], h2[-1]):.4f}")
print(f"\n  [Bidi, full-context]  cos(h_t) = {cos(encode_dialogo(dialogo_confirma, f2_bidi)[1][-1], encode_dialogo(dialogo_cancela, f2_bidi)[1][-1]):.4f}")
"""),
    md(r"""
**Lectura.** El static no puede distinguir (`"sí"` es `"sí"`, mismo vector → cos = 1.0000). El
**contextual AR** sí, y **fuerte**: el mismo turno se separa (cos ≈ 0.57) según si confirma una reserva
o acepta una cancelación. (El **Bidi** acá separa mucho menos, cos ≈ 0.95 — su geometría queda más
pegada al turno.) Esto es lo que aporta el contextual y un embedding por-turno aislado no puede capturar.
"""),
    md(r"""
## Demo 2 — Memoria conversacional: ¿qué situación se parece más?

Una **consulta** (turno + su contexto) y un set de **situaciones candidatas**. Recuperamos la más
parecida por coseno, comparando static vs contextual. Score más alto = más parecida.
"""),
    code(r"""
consulta = [
    ("user", "Hace una semana que mi pedido figura 'en camino' y no llega."),
    ("system", "Lamento la demora. ¿Querés que lo revise?"),
    ("user", "Sí, y la verdad quiero que me devuelvan la plata."),
]
candidatas = {
    "reclamo / reembolso": [
        ("user", "Me cobraron dos veces la suscripción este mes."),
        ("system", "Veo el doble cargo. ¿Cómo procedo?"),
        ("user", "Quiero que me reintegren uno de los cobros."),
    ],
    "saludo inicial": [
        ("user", "Hola, buen día."),
        ("system", "¡Hola! ¿En qué te ayudo?"),
        ("user", "Quería hacer una consulta."),
    ],
    "reserva": [
        ("user", "Quiero una mesa para dos el viernes."),
        ("system", "¿A qué horario?"),
        ("user", "A las nueve de la noche."),
    ],
}

def repr_last(dialogo, contextual):
    e, h = encode_dialogo(dialogo, f2_ar)
    return (h if contextual else e)[-1]

q_e, q_h = repr_last(consulta, False), repr_last(consulta, True)
print("Consulta: usuario reclama por un pedido demorado y pide reembolso\n")
print(f"{'candidata':<22}{'static (e_t)':>14}{'contextual (h_t)':>18}")
for name, dlg in candidatas.items():
    se = cos(q_e, repr_last(dlg, False))
    sh = cos(q_h, repr_last(dlg, True))
    print(f"{name:<22}{se:>14.4f}{sh:>18.4f}")
"""),
    md(r"""
**Lectura honesta.** Recuperación a nivel **situación**, con scores. Acá el **static ordena mejor**
(pone *reclamo/reembolso* primero), mientras el **contextual** zero-shot levanta *saludo inicial*: la
geometría aprendida **todavía no está calibrada para retrieval de situación** — exactamente lo que
mostró la MSS@10. La **disambiguación** (Demo 1) sí funciona; la **similitud-de-situación** es lo que
falta afinar (objetivo contrastivo a nivel situación). Demo cualitativo sobre diálogos hechos a mano.

> El número formal (act-match P@k, MSS@10 con juez LLM, reconciliación con el paper) está en
> `packages/conversational-ann/results/RESULTS.md`.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
path = NB_DIR / "demo_turnos_contextuales.ipynb"
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("escrito", path)
