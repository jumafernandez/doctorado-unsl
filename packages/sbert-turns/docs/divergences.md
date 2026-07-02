# SBERT-de-turnos — Registro de divergencias

El artefacto **sbert-turns** agrega a TRACE dos tokens especiales `[CLS]`/`[SEP]` **estilo RoBERTa** para tener
un slot que resuma el diálogo. `SBertTurnModel` **subclasa** `ContextualTurnModelV2` (paquete hermano
`contextual-turn-embeddings`) y el pipeline **importa/reusa** ese paquete — **no lo edita**. Hereda **todas** las
divergencias de v2 (ver `contextual-turn-embeddings/docs/model/v2.md`) y agrega las de abajo.

## En simple (TL;DR)

Le ponemos a TRACE los dos tokens de BERT (`[CLS]` al inicio, `[SEP]` entre diálogos) como **dos vectores
aprendibles**, empaquetamos varios diálogos por secuencia (como RoBERTa llena el contexto con oraciones), y
**entrenamos con la misma loss de siempre**. La clave (idea de Sergio): a diferencia de BERT —que entrena el
`[CLS]` con NSP/NLI— **RoBERTa mostró que no hace falta un objetivo para el CLS**; lo dejamos viajar de rebote.
Nos sirve porque **no tenemos** una tarea no supervisada para ese token con diálogos (todavía).

| | BERT / RoBERTa (palabras) | SBERT-de-turnos |
|---|---|---|
| **Vocabulario de especiales** | filas de `word_embeddings` en el vocab | **`nn.Embedding(2, D)`** — solo CLS y SEP |
| **Objetivo del CLS** | BERT lo entrena (NSP/NLI) | **ninguno** (RoBERTa-style; viaja de rebote) |
| **Unidad empaquetada** | oraciones hasta llenar el contexto | **diálogos**: `[CLS] d1 [SEP] d2 [SEP] …`, 1..n variable |
| **Objetivo de entrenamiento** | MLM | **el mismo de TRACE** (masked-recon en bidi) |

> **Principio:** igual que el v2, lo que no hace falta cambiar se deja igual y **toda** divergencia se registra
> acá. Lo que **no** está en este paso: SimCSE ni objetivos a nivel diálogo (fase posterior).

## Divergencias (además de las de v2)

| # | Qué | BERT/RoBERTa | SBERT-de-turnos | Por qué |
|---|---|---|---|---|
| S1 | Tokens especiales | `[CLS]`/`[SEP]` = filas del vocab de palabras | **`nn.Embedding(2, D)`** aprendible (fila 0=CLS, 1=SEP); el SEP se repite pero es **un solo vector** | no hay vocab de turnos; solo se necesitan esos dos |
| S2 | Espacio de los especiales | en `hidden` (word_embeddings) | en **`input_dim`** (como `e_t`) → pasan por el mismo `input_proj`+posición que los turnos | tratarlos idénticos a un turno de entrada |
| S3 | Objetivo del CLS | BERT: NSP/NLI entrena el CLS | **ninguno** — se sustituyen en la secuencia y se entrenan **de rebote** por la loss existente | RoBERTa mostró que no hace falta; no hay tarea no-supervisada de diálogo (aún) |
| S4 | Entrada | una secuencia por ejemplo | **packing**: `[CLS] d1 [SEP] d2 [SEP] …`, **1..n diálogos** hasta llenar `max_turns` | RoBERTa "full-sentences", a nivel diálogo |
| S5 | Masking / targets | MLM enmascara tokens (incl. contexto) | **CLS/SEP nunca se enmascaran ni son target** (se usa `turn_mask = attention_mask & (special_ids==0)`) | son estructura, no contenido a reconstruir |
| S6 | Fronteras next-turn | n/a | el `turn_mask` **corta en el SEP** (el t+1 del último turno de un diálogo es un SEP → inválido) → sin cruces entre diálogos | un turno no "predice" el de otro diálogo |
| S7 | Segmento / token_type | token_type A/B en CLS/SEP | speaker **"unknown"** (`num_speakers-1`) en CLS/SEP | evita tocar `BertTurnEmbeddings`; BERT igual les da un token_type (fiel-ish) |
| S8 | Atención entre segmentos | RoBERTa: atención plena a través de fronteras del pack | **igual** — SEP es solo un marcador, sin máscara de segmento | fiel a RoBERTa (packing con atención cruzada) |
| S9 | Dirección | BERT es bidireccional | **solo bidi** en este paso (el CLS al inicio solo resume con atención bidireccional; AR con CLS-al-final = variante posterior) | fiel: SBERT es BERT bidireccional |
| S10 | Posiciones | CLS en pos 0 | `arange` sobre el pack, **CLS en pos 0** (fiel); restricción: pack ≤ `max_turns` (tamaño de la tabla de posiciones) | mismo mecanismo de BERT |

## Persistencia y compatibilidad

`SBertTurnModel` hereda `save_pretrained`/`from_pretrained`/`encode` de `ContextualTurnModelV2`; el
`state_dict` ya incluye `special_embeddings` y `from_pretrained` usa `cls(config)` (funciona en la subclase).
Un `ContextualTurnModelV2` normal **no** tiene `special_embeddings` (verificado en `tests/test_sbert_turns.py`).
