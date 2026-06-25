# papers/

Writeups (LaTeX, formato **ACL 2-columnas** — el de los venues de Sergio: \*ACL / SIGDIAL).
El código y los experimentos viven en `../packages/`; acá va **solo la prosa**.

| dir | paper | estado |
|---|---|---|
| `contextual-turn-encoder/` | **Nuestro BERT** — encoder contextual de turnos pre-entrenado para TOD (act-trajectory). Resultados desde `packages/contextual-turn-embeddings/benchmarks/`. | activo (esqueleto + tabla held-out + `% TODO`) |
| `ann-memory/` | Embeddings dinámicos + ANN para recuperación de memoria. Código/experimentos en `../packages/conversational-ann/` y el repo `ANN-UNSL`. | esqueleto |

## Compilar

Cada dir es autocontenido (trae `acl.sty` + `acl_natbib.bst`, oficiales de
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)):

```bash
cd contextual-turn-encoder
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`[review]` en `\usepackage[review]{acl}` da numeración de líneas (borrador). Para camera-ready, quitarlo.

## Pendientes que el esqueleto ya marca (para consolidar)
- **significancia** (bootstrap/Wilcoxon sobre diálogos)
- **held-out limpio como LA tabla** (no la transductiva) — ya está en `tab:main`
- **decidir la historia del modelo** (presentar UNO + ablación, no el lineage v1/v2/v3)
- **related work** + **breakdown por dataset** + fila **act(t)** de control
