# Entrenamiento

El entrenamiento de `f2` está en `train.train(config, df=None, embeddings=None,
base_encoder=None, verbose=True)`. También se puede lanzar por CLI:

```bash
python scripts/train_contextual_turn_model.py --config configs/default.yaml
```

## Flujo

```text
1. set_seed + get_device                         # reproducibilidad y dispositivo (auto/cpu/cuda/mps)
2. cargar datos                                  # df explícito o config.data.path -> load_dataframe
3. normalize_columns                             # columnas canónicas + row_id
4. resolver embeddings base e_t                  # arg embeddings > columna 'embedding' > BaseTurnEncoder
5. _prepare_model_config                         # infiere input_dim/output_dim de e_t reales
6. resolve_losses_for_mode                       # defaults por modo + advertencias leaky
7. DialogueDataset + DataLoader (collate)        # batches de diálogos con padding
8. AdamW + scheduler warmup lineal               # build_linear_warmup_scheduler
9. loop por época:
     - mover batch al device
     - (opcional) enmascarar turnos
     - forward(s) de f2
     - compute_objectives -> total ponderado
     - backward, clip de gradiente, step, scheduler.step
     - logging por paso
10. guardar checkpoint + logs + config           # save_pretrained + training_log.jsonl + config.yaml
```

## Qué se guarda

En `config.training.output_dir`:

```text
config.json            # ModelConfig (para from_pretrained)
model.safetensors      # pesos
training_args.json     # training + losses + device + total_steps + timestamp
config.yaml            # la configuración completa, ya resuelta
training_log.jsonl     # una línea por paso: epoch, step, lr y cada componente de la pérdida
```

> `output_dir` debería apuntar a una carpeta ignorada por git (p. ej. `models/` o `outputs/`,
> ambas en `.gitignore`). No se versionan checkpoints ni logs de corridas.

## Detalles de implementación

- **Masking**: solo si `masked_reconstruction` (o `embedding_retrieval` con `target=masked`) está
  activo; usa `mask_prob` de `masked_reconstruction`.
- **Forwards**: a lo sumo dos por batch — uno sobre la secuencia enmascarada (lado masked) y uno
  sobre la secuencia limpia (lado next-turn). Los objetivos del mismo lado **reusan** ese forward.
- **Pérdida total**: suma ponderada por los `weight` de cada objetivo activo.
- **Gradiente**: `clip_grad_norm_` con `gradient_clip_norm`.
- **Mixed precision**: `mixed_precision=True` solo tiene efecto en CUDA; en CPU se ignora de forma
  segura.
- **Logging**: cada paso se registra en memoria y se vuelca a `training_log.jsonl`. La consola
  imprime cada `log_interval` pasos (si `verbose`).

## Bidireccional vs autoregresivo

| | bidireccional | autoregresivo |
|---|---|---|
| Atención | todos los turnos (padding enmascarado) | solo `j ≤ t` (máscara causal) |
| Objetivo primario | `masked_reconstruction` | `next_turn_prediction` |
| `embedding_retrieval` (`auto`) | sobre posiciones enmascaradas | sobre próximos turnos válidos |
| Uso típico | diálogo completo, contexto rico | embeddings online/streaming |

## Configuraciones iniciales recomendadas

- **Bidireccional (default):** `masked_reconstruction` ON, `next_turn_prediction` OFF. Buen punto
  de partida para representaciones de diálogo completo.
- **Autoregresivo:** `next_turn_prediction` ON; opcionalmente `masked_reconstruction` como
  auxiliar. Para representaciones causales.
- **Con retrieval:** activá `embedding_retrieval` (`enabled: true`, `target: auto`) además del
  objetivo primario; empezá con `temperature: 0.07`, `weight: 1.0`. Ver
  [configuration.md](configuration.md).

Ver el archivo `configs/default.yaml` como base y [configuration.md](configuration.md) para cada
campo.

## Progresión experimental sugerida

```text
1. smoke test con datos de juguete        # valida la implementación (CPU, segundos)
2. subconjunto chico de Dialog2Flow       # primera corrida real, pocas épocas
3. diagnósticos de contextualidad         # ¿usa el contexto de forma medible?
4. corrida de entrenamiento más grande    # más diálogos / épocas
5. evaluación ANN/MSS                      # validación científica (etapa posterior)
```

> **Importante.** Que pasen el smoke test y `pytest` valida la **implementación**, no una mejora
> científica. La superioridad downstream se decide recién en el paso 5 (ANN/MSS), fuera del
> alcance de este paquete. Ver [research_notes.md](research_notes.md).
