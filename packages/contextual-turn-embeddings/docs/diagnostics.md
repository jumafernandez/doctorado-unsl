# Diagnósticos de contextualidad

El notebook `notebooks/colab_d2f_contextuality_diagnostics.ipynb` (pensado para Google Colab,
texto en español) evalúa si los embeddings contextuales **realmente usan el contexto** de forma
medible, sobre un subconjunto chico de Dialog2Flow con embeddings precomputados. **No** entrena un
modelo nuevo por defecto: carga el modelo *smoke* previo.

> **Importante.**
> ```text
> Estos diagnósticos no prueban superioridad downstream en recuperación ANN/MSS.
> Sólo verifican si el modelo usa el contexto de manera medible y no caótica.
> ```

Convenciones: `e_t` = embedding base, `h_t` = embedding contextual (contexto real). El
subconjunto preserva **diálogos completos** y descarta diálogos demasiado cortos (sin contexto)
mediante `MIN_DIALOGUE_TURNS`.

## Diagnóstico 1 — Reconstrucción enmascarada: contexto real vs. alterado

**Qué testea.** Para cada diálogo se enmascara un turno objetivo y se mide la pérdida de
reconstrucción bajo cuatro contextos: real, mezclado dentro del diálogo
(`shuffled_within_dialogue`), aleatorio de otros diálogos (`random_cross_dialogue`) y sin contexto
(`no_context`).

- **Patrón bueno:** `loss(real) < loss(shuffled) , < loss(random) , < loss(no_context)`. El
  contexto real ayuda más que cualquier contexto corrupto.
- **Patrón malo:** el contexto real **no** es el de menor pérdida → el modelo no estaría usando la
  estructura conversacional de forma significativa (el notebook lo advierte en español).
- **Patrón ambiguo:** `real` mejor que `no_context` pero similar a `shuffled`/`random` → reacciona
  a *tener* contexto, pero es poco sensible a *cuál* es.
- **No prueba:** mejor retrieval downstream.

Es el diagnóstico **más importante**.

## Diagnóstico 2 — Sensibilidad del embedding contextual

**Qué testea.** Para los mismos turnos objetivo (sin enmascarar) compara `h_t` bajo distintos
contextos: `cos(h_real, h_shuffled)`, `cos(h_real, h_random)`, `cos(h_real, h_no_context)` y
`cos(e_base, h_real)` (media/mediana/desvío/p10/p90 + histogramas).

- **Patrón bueno:** cambios **medibles pero no caóticos** al cambiar el contexto.
- **Patrón malo (insensible):** todas las similitudes ≈ 1 → el contexto casi no cambia `h_t`.
- **Patrón malo (caótico):** similitudes muy bajas / `cos(e_base, h_real)` ≈ 0 → el modelo
  sobre-transforma o desestabiliza el espacio.
- **No prueba:** que la sensibilidad se traduzca en mejor recuperación.

## Diagnóstico 3 — Misma utterance, distinto contexto

**Qué testea.** Para utterances repetidas (normalizadas, con ≥ `MIN_REPETITIONS` apariciones; p.
ej. "sí", "ok", "gracias") compara la similitud media par-a-par entre sus embeddings **base** vs
**contextuales**. Define `dispersión contextual = sim_base − sim_contextual`.

- **Patrón bueno:** dispersión > 0 → mismo texto en distintos contextos se **separa** más en el
  espacio contextual.
- **Patrón malo:** dispersión ≤ 0 → el contexto no separa las repeticiones (o las junta más).
- **Patrón ambiguo:** dispersión cercana a 0.
- **No prueba:** que esa separación sea la "correcta" funcionalmente.

## Diagnóstico 4 — Inspección cualitativa de vecinos exactos

**Qué testea.** Para unas pocas consultas, compara los vecinos más cercanos (coseno **exacto**,
sin FAISS/ANN) recuperados por `e_t` vs por `h_t`, evitando vecinos del mismo diálogo.

- **Patrón bueno:** los vecinos contextuales muestran **menos repetición léxica** y situaciones
  funcionalmente más plausibles que los base.
- **Patrón malo:** sin diferencia cualitativa, o vecinos contextuales sin sentido.
- **Patrón ambiguo:** mezcla; difícil de juzgar a ojo.
- **No prueba:** nada cuantitativo; es **cualitativo** y sobre un subconjunto chico. No es la
  evaluación ANN/MSS final.

## Diagnóstico 5 — Resumen del cambio geométrico

**Qué testea.** `cos(e_t, h_t)`, normas de `e_t` y `h_t`, y `Overlap@10(base, contextual)` (solapa
de los 10 vecinos exactos por coseno, sobre una muestra).

- **Patrón bueno:** overlap **intermedio** → reorganización contextual significativa pero no
  destructiva; `cos(e_t, h_t)` en un rango razonable (cambia, sin colapsar a 0).
- **Patrón malo (apenas cambia):** overlap ≈ 1, `cos(e_t, h_t)` ≈ 1.
- **Patrón malo (destruye geometría):** overlap ≈ 0.
- **No prueba:** que el cambio geométrico mejore la tarea final.

## Checklist e interpretación

El notebook cierra con un checklist en español que marca automáticamente lo que es verificable
(contexto real mejor; `h_t` cambia con el contexto; cambio no caótico; utterances genéricas se
separan; sin NaNs; diagnósticos guardados) y deja la inspección de vecinos para revisión manual.

Salidas (en un directorio de resultados, ignorado por git): `diagnostic_losses_by_context.csv`,
`context_sensitivity_summary.csv`, `repeated_utterance_contextual_dispersion.csv`,
`qualitative_neighbors_base_vs_contextual.csv`, `geometry_shift_summary.csv`,
`diagnostic_config.json`, y tres PNG. Ver el notebook para detalles operativos (rutas de Drive,
selección del subconjunto, etc.).

> Reiteramos: la validación final requiere la evaluación posterior ANN/MSS cross-dialogue
> comparando Static, Dynamic, EMA y las variantes Contextual. Ver
> [research_notes.md](research_notes.md).
