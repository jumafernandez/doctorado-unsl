# sbert-turns

**Sentence-BERT de diálogos**: agrega los tokens especiales `[CLS]`/`[SEP]` de BERT —**estilo RoBERTa**— a un
encoder contextual de turnos (TRACE / `ContextualTurnModelV2` del paquete hermano `contextual-turn-embeddings`)
para tener un slot que **resuma el diálogo**.

Es un **artefacto separado** que reusa el paquete base por **import/subclase**, sin editarlo. Idea de Marcelo
y Sergio; primer paso de una línea "SBERT de turnos" (las tareas a nivel diálogo para explotar el CLS vienen
después).

## Idea (paso 1)

- Entrada empaquetada: `[CLS] diálogo1 [SEP] diálogo2 [SEP] …` — packing variable de 1..n diálogos hasta llenar
  el contexto (RoBERTa "full-sentences", a nivel diálogo).
- `[CLS]`/`[SEP]` = **dos vectores aprendibles** (`nn.Embedding(2, D)`). El SEP se repite pero es un solo vector.
- **Sin objetivo propio para el CLS** (RoBERTa mostró que no hace falta): se entrena **de rebote** con la misma
  loss de TRACE (masked-recon en bidi). Las posiciones CLS/SEP **nunca** se enmascaran ni son target.
- Variante **bidi** (el CLS al comienzo solo resume con atención bidireccional).

## Componentes

- `sbert_turns/model.py` — `SBertTurnModel(ContextualTurnModelV2)`: subclase + `nn.Embedding(2, D)`.
- `sbert_turns/data.py` — `PackedDialogueDataset` + `collate_packed` (packing con CLS/SEP).
- `sbert_turns/objective.py` — `compute_objectives_sbert` (la loss de siempre, excluye CLS/SEP con `turn_mask`).
- `training/train_sbert.py` — recipe headless (v3-bidi, o `--recipe v2` barato).
- `docs/divergences.md` — registro de divergencias vs BERT/RoBERTa.
- `tests/` — tests download-free.

## Instalación

```bash
# desde la raíz del monorepo, con el venv de contextual-turn-embeddings:
pip install -e packages/contextual-turn-embeddings   # base (TRACE) — si no está
pip install -e packages/sbert-turns                  # este artefacto
pytest packages/sbert-turns
```

## Entrenar

```bash
python packages/sbert-turns/training/train_sbert.py --recipe v3 --epochs 15   # v3-bidi sobre d2f-full
python packages/sbert-turns/training/train_sbert.py --recipe v2 --epochs 8    # sanity rápido (6 capas)
```
