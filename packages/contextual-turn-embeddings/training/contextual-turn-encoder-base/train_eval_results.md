# v1 / v2 / v3 — comparación de entrenamiento (full)

**(2026-06-21)** Sobre el corpus **full** (~1.97M turnos), mismo `f1`, mismo held-out (semilla 42), mismo
objetivo. Tres modelos (arquitectura × receta × tamaño):

- **v1** — custom (pre-LN + residual), receta propia (`lr 2e-4`), 6 capas/8 heads.
- **v2** — **BERT-fiel** (post-LN), receta del v1 (`lr 2e-4`), 6 capas/8 heads.
- **v3** — **BERT-base literal** (12 capas/12 heads/3072), receta de BERT (`lr 1e-4`, no-decay en bias/LN).

## Curvas

**v1 / v2 / v3 — comparación (val sólido, train punteado):**

![comparación full](figures/v1_vs_v2_full_curves.png)

**v2 / v3 (train + val por modo):**

![curvas BERT-fiel](figures/v2_full_curves.png)

## Eval loss (best-by-val)

| modo | v1 | v2 | v3 (BERT-base) |
|---|---|---|---|
| **AR** | 4.676 (ep4) | **4.278** (ep15) | 4.416 (ep5) |
| **Bidi** | 2.952 (ep4) | **2.845** (ep11) | 2.965 (ep15) |

## Lectura

- **El v2 gana el proxy en los dos modos.** El v3 (BERT-base literal) quedó **entre v1 y v2 en AR** y
  **≈ v1 en Bidi**: le gana al custom v1, pero **no le gana al v2**.
- **AR:** el v3 (12 capas) **overfittea rapidísimo** — best en ep5, después el val sube (como el v1).
  Más capacidad → sobreajuste más temprano.
- **Bidi:** el v3 con `lr=1e-4` **converge lento** y **seguía bajando despacio en ep15** (no terminó de
  converger). Parte del gap es **sub-entrenamiento**: el lr de BERT está pensado para corpus gigantes y
  es lento a esta escala; cortamos en 15 (límite de cómputo).
- **train vs val:** v2 y v3 ajustan el train casi igual de fuerte (≈0.94 AR / ≈0.27 Bidi); el v3 **no
  generaliza mejor** pese a ser más grande.

## Conclusión

Escalar a **BERT-base de manual + receta de BERT (v3) NO movió la aguja** sobre el proxy frente al v2
más chico y mejor afinado. El v3 es el **baseline simil-BERT fiel** pedido — y su lectura es clara: el
**lever no es el tamaño ni la receta, es el OBJETIVO**. El próximo salto va por la **Fase 2 (codebook /
clasificación)**, no por más capas.

> **Caveat de siempre:** esto es el **proxy** auto-supervisado, **no** el veredicto. El **downstream**
> (act-match / MSS@10) puede dar otra cosa — está **pendiente** para v2 y v3 (encodear la colección 1M
> con cada checkpoint y correr `eval_prelim`).

## Notas de las corridas

- **v3-AR:** entrenado en **Colab/GPU** (`lr 1e-4`, ~19 min/época). Convergió en **ep5** (best 4.416) y
  Colab se desconectó en ep14 — irrelevante, todo post-ep5 era overfit. Curva reconstruida del log
  (ep1–13); el checkpoint `best/` está en **Drive** (bajar para el eval downstream).
- **v3-Bidi:** entrenado en la **M2** (`mps`, ~2 h/época, 15 épocas). `best/` local (ep15).

## Reproducibilidad

- **v1:** [`02_train_contextual_m2.ipynb`](02_train_contextual_m2.ipynb) · **v2:** [`03_…`](03_train_contextual_v2_m2.ipynb)
  · **v3 (M2):** [`04_…`](04_train_contextual_v3_m2.ipynb) · **v3 (Colab/GPU):** [`05_…`](05_train_contextual_v3_colab.ipynb).
- **Curvas:** [`plot_full_results.py`](plot_full_results.py) (version-aware: detecta v1/v2/v3 solo).
