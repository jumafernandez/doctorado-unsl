# Objetivos de entrenamiento (losses)

El paquete entrena `f2` con objetivos **auto-supervisados** (no requieren etiquetas
funcionales). Hay tres, todos opcionales y configurables; los dos primeros comparten un término
base de MSE + coseno.

| Objetivo | Clave en config / resultados | Modo primario |
|----------|------------------------------|---------------|
| Reconstrucción enmascarada | `masked_reconstruction` | bidireccional |
| Predicción del próximo turno | `next_turn_prediction` | autoregresivo |
| Recuperación de embeddings | `embedding_retrieval` | ambos (`target=auto`) |

Convención de shapes: ver [architecture.md](architecture.md). `B,S` = batch y turnos;
`D = D_in` salvo aclaración. Todas las funciones son **empty-safe**: si no hay posiciones
seleccionadas, devuelven un cero **diferenciable** (no rompen el loop ni aportan gradiente).

## Término base: `mse_cosine_loss`

```text
mse_cosine_loss(pred [N, D], target [N, D], lambda_cosine=1.0)
  = MSE(pred, target) + lambda_cosine * mean(1 - cos(pred, target))
```

Combina error cuadrático (magnitud/posición) y distancia coseno (dirección). `N` son las
posiciones ya seleccionadas (aplanadas). Si `N == 0`, devuelve 0.

## 1. Reconstrucción enmascarada (`masked_reconstruction`)

Análogo del **masked language modeling**, pero a nivel de turno.

Procedimiento:

1. `apply_turn_masking(embeddings, attention_mask, mask_prob, mask_embedding)` elige al azar
   algunos **turnos válidos** (prob. `mask_prob`, def. `0.15`) y reemplaza su `e_t` por un vector
   aprendido `mask_embedding`. Devuelve `mask_positions [B, S]` (solo turnos reales).
2. Se corre `f2` sobre la secuencia enmascarada → `hidden [B, S, D_out]`.
3. `reconstruction_head(hidden)` reconstruye el `e_t` original en espacio `D_in`.
4. La pérdida solo cuenta las posiciones enmascaradas:

```text
masked_reconstruction_loss(pred [B,S,D], target=e_t [B,S,D], mask_positions [B,S], lambda_cosine)
  = mse_cosine_loss(pred[mask_positions], e_t[mask_positions], lambda_cosine)
```

Es el objetivo primario en **bidireccional**: para reconstruir un turno oculto el modelo debe
inferirlo de los **vecinos** (pasados y futuros), lo que fuerza a `h_t` a integrar contexto.

> Variante simplificada: siempre se reemplaza por el vector `[MASK]` (sin el truco 80/10/10 de
> BERT). Es una decisión deliberada a nivel de embeddings.

## 2. Predicción del próximo turno (`next_turn_prediction`)

Desde `h_t`, predecir el embedding **base** del próximo turno `e_{t+1}`.

```text
targets, valid = build_next_turn_targets(e_t [B,S,D], attention_mask [B,S])
   targets[:, t] = e_t[:, t+1]
   valid[:, t]   = (turno t real) AND (turno t+1 real)   # excluye padding y último turno
hidden = f2(e_t, attention_mask, speaker_ids)            # forward limpio (sin enmascarar)
pred_next = next_turn_head(hidden)
next_turn_prediction_loss(pred_next, targets, valid, lambda_cosine)
  = mse_cosine_loss(pred_next[valid], targets[valid], lambda_cosine)
```

Es el objetivo primario en **autoregresivo**. El último turno de cada diálogo (sin "próximo") y
el padding se ignoran vía `valid`.

### Por qué es leaky en bidireccional

En bidireccional, `h_t` atiende a **todos** los turnos, incluido `t+1`. Predecir `e_{t+1}` desde
un `h_t` que ya "vio" `t+1` es casi trivial (fuga de información) y empuja a `h_t` a codificar el
futuro. Por eso este objetivo es primario solo en autoregresivo; activarlo en bidireccional emite
una advertencia (ver "Defaults por modo").

## 3. Recuperación de embeddings (`embedding_retrieval`)

El objetivo nuevo y opcional (apagado por defecto). Es el **análogo a nivel de turno de la
proyección de vocabulario** de un modelo de lenguaje.

```text
Modelo de lenguaje:
    h_t @ W_vocab.T          → logits sobre tokens del vocabulario

Modelo a nivel de turno:
    h_t @ E_candidates.T     → scores sobre embeddings de turnos candidatos
```

Tratamos los embeddings de turnos como un "vocabulario de turnos" y pedimos que `h_t` puntúe alto
al turno objetivo correcto frente a otros candidatos. Se entrena como **retrieval contrastivo
in-batch**:

```text
embedding_retrieval_loss(Q [M, D], T [M, D], temperature=0.07, normalize=True):
    Q = L2norm(Q) if normalize else Q
    T = L2norm(T) if normalize else T
    scores = (Q @ T.T) / temperature        # [M, M]
    labels = [0, 1, ..., M-1]                # la diagonal es el positivo
    loss   = cross_entropy(scores, labels)
```

donde:

- `Q` = embeddings contextuales de consulta (los `h_t`),
- `T` = embeddings base objetivo (positivos, alineados fila a fila con `Q`),
- `M` = cantidad de objetivos en el batch.

Sketch de la matriz de scores (la diagonal son los positivos; el resto, negativos in-batch):

```text
        T_0   T_1   T_2   ...
 Q_0 [  +     -     -    ]
 Q_1 [  -     +     -    ]
 Q_2 [  -     -     +    ]
  ...
```

### Negativos in-batch (no softmax sobre todo el corpus)

Los **negativos** son los demás objetivos del **mismo batch**: cada query debe elegir su positivo
(diagonal) entre los `M-1` restantes. Es una aproximación barata y escalable al softmax completo
sobre todo el corpus, que sería inviable (p. ej. ~1M embeddings). v1 implementa **solo** este
modo (`candidate_mode="in_batch"`).

### Wrappers según el modo (`target=auto`)

El "objetivo" (`target`) define qué son `Q` y `T`; con `target=auto` se resuelve según el modo de
atención dentro de `compute_objectives`:

- **bidireccional → `masked`** (`masked_embedding_retrieval_loss`): `Q = hidden[mask_positions]`
  (los `h_t` de los turnos enmascarados), `T = e_t[mask_positions]` (sus embeddings base
  originales). Reusa el forward enmascarado de la reconstrucción.

  ```text
  masked_embedding_retrieval_loss(hidden [B,S,D], base=e_t [B,S,D], mask_positions [B,S], temperature, normalize)
  ```

- **autoregresivo → `next_turn`** (`next_turn_embedding_retrieval_loss`): `Q = hidden[valid]`,
  `T = targets[valid]` (los `e_{t+1}`). Reusa el forward causal de próximo turno.

  ```text
  next_turn_embedding_retrieval_loss(hidden [B,S,D], next_targets [B,S,D], valid_mask [B,S], temperature, normalize)
  ```

`target` también puede fijarse explícitamente en `masked` o `next_turn`.

### Query: `h_t` crudo vs head existente

- Si `D_out == D_in` (default), la query es el **`h_t` crudo** (el fin es que el embedding
  contextual *en sí* sea discriminativo).
- Si `D_out != D_in`, se reusa la head existente como proyección (`reconstruction_head` para
  `masked`, `next_turn_head` para `next_turn`) para llevar `h_t` al espacio de los candidatos.
- En ningún caso se agregan parámetros nuevos al modelo: el checkpoint sigue siendo compatible.

### Falsos negativos

Como los negativos son in-batch sin filtrar, dos turnos con **texto genérico repetido** ("sí",
"gracias", "ok") dentro del mismo batch pueden tratarse como negativos siendo casi idénticos
(falsos negativos). Es una limitación aceptada en v1.

### Extensiones futuras

- negativos muestreados (sampled negatives);
- memory bank de embeddings;
- hard negatives con FAISS;
- una head de retrieval sobre todo el corpus.

Ver [research_notes.md](research_notes.md).

## Defaults por modo y advertencias (`resolve_losses_for_mode`)

`resolve_losses_for_mode(loss_config, attention_mode)` resuelve los objetivos cuyo `enabled` está
en `None` según el modo, y deja intactos los `True/False` explícitos:

- **bidireccional** → `masked_reconstruction` ON, `next_turn_prediction` OFF.
- **autoregresivo** → `next_turn_prediction` ON, `masked_reconstruction` OFF (queda como auxiliar
  opcional si se la activa explícitamente).

Emite un `UserWarning` claro en dos combinaciones **leaky**:

1. `next_turn_prediction` activado en **bidireccional**;
2. `embedding_retrieval` con `target="next_turn"` en **bidireccional**.

`embedding_retrieval.enabled` es un booleano explícito (def. `False`); no se resuelve por modo. La
resolución de su `target=auto` ocurre en `compute_objectives`.

## Cómo se combinan (`compute_objectives`)

`compute_objectives(model, batch, loss_config, generator=None)` devuelve un dict con las claves de
los objetivos **activos** (`masked_reconstruction`, `next_turn_prediction`, `embedding_retrieval`)
y una clave `total` = suma ponderada por los `weight` de cada objetivo. Reusa el forward
enmascarado y el forward limpio entre objetivos del mismo "lado" (no agrega pasadas extra). Las
salidas son escalares diferenciables.

> La clave de resultado/log del objetivo nuevo es `embedding_retrieval` (consistente con las otras
> claves), no `loss_embedding_retrieval`.
