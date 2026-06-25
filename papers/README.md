# papers/

Writeups (LaTeX). El código y los experimentos viven en `../packages/`; acá va **solo la prosa**.
Cada paper en el **formato de su venue target** y es self-contained (trae sus `.cls/.sty/.bst`).

| dir | paper | formato | main | estado |
|---|---|---|---|---|
| `contextual-turn-encoder/` | **Nuestro BERT** — encoder contextual de turnos pre-entrenado para TOD (act-trajectory). Resultados desde `packages/contextual-turn-embeddings/benchmarks/`. | **ACL 2-col** (target: SIGDIAL / \*ACL, las ligas de Sergio) | `main.tex` | esqueleto + tabla held-out + `% TODO` |
| `ann-memory/` | **"Retrieving the Flow"** — representaciones dinámicas de turno + ANN para memoria conversacional. Código en `../packages/conversational-ann/` + repo `ANN-UNSL`. | **LNCS 1-col** | `paper-main-en.tex` (`-es` = castellano) | **escrito** (20 págs, 29 refs) |

## Compilar

```bash
# BERT (ACL):
cd contextual-turn-encoder && pdflatex main && bibtex main && pdflatex main && pdflatex main
# ANN (LNCS):
cd ann-memory && pdflatex paper-main-en && bibtex paper-main-en && pdflatex paper-main-en && pdflatex paper-main-en
```
Los `.pdf` (compilados) e intermedios están gitignoreados; las figuras `.pdf` **no** (son fuente).
En el BERT, `[review]` en el preámbulo da numeración de líneas; quitarlo para camera-ready.

## Pendientes del paper del BERT (para consolidar)
- **significancia** (bootstrap/Wilcoxon sobre diálogos)
- **held-out limpio como LA tabla** (no la transductiva) — ya está en `tab:main`
- **decidir la historia del modelo** (presentar UNO + ablación, no el lineage v1/v2/v3)
- **related work** + **breakdown por dataset** + fila **act(t)** de control

> El ANN viene de Overleaf. Los drafts viejos (`versiones_anteriores/`, `figures/prev/`) quedaron
> **fuera del repo** a propósito (siguen en el zip original en `~/Downloads/paper-ann`).
