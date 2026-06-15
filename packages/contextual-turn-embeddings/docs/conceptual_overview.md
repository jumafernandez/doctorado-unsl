# Panorama conceptual

Este documento explica la **idea de investigación** detrás del paquete, sin entrar en
detalles de implementación (para eso, ver [architecture.md](architecture.md) y
[contextual_model.md](contextual_model.md)).

## 1. Contextualización a nivel de token vs. a nivel de turno

Los modelos de lenguaje modernos contextualizan **tokens**: la representación de una palabra
depende de las palabras que la rodean. En un encoder tipo BERT, el embedding de un token se
calcula atendiendo a todos los tokens de la secuencia; en un decoder tipo GPT, atendiendo solo
a los tokens anteriores.

Nuestra unidad de interés no es el token sino el **turno de diálogo** (la intervención completa
de un hablante). Queremos que la representación de un turno dependa de los **otros turnos** de
la conversación. Es decir, trasladamos la idea de contextualización **un nivel hacia arriba**:

```text
LLM clásico:     token  contextualizado por otros tokens
Este paquete:    turno  contextualizado por otros turnos
```

Por eso es útil pensar en `ContextualTurnModel` como "un BERT/GPT sobre turnos": el
`TransformerEncoder` opera sobre una secuencia cuyas posiciones son turnos, y cada turno se
representa con un vector (su embedding base) en lugar de un id de vocabulario.

## 2. Por qué un embedding base `e_t` puede ser insuficiente

Un **embedding base** `e_t` codifica un turno **de forma aislada**: solo depende del texto de
ese turno. Es lo que produce cualquier *sentence encoder* (p. ej. SentenceTransformers).

El problema es que en un diálogo el **significado funcional** de un turno depende del contexto.
Ejemplos:

- "sí" puede ser una confirmación de reserva, una respuesta a "¿querés algo más?", o un
  acuse de recibo. El texto es idéntico; la función conversacional, no.
- "a las 8" solo se entiende si antes se preguntó por un horario.

Un embedding base asigna **el mismo vector** a todas las apariciones de "sí", borrando esa
distinción. Para memoria conversacional y recuperación, esto mezcla situaciones que deberían
estar separadas.

## 3. Qué representa un embedding contextual `h_t`

Un **embedding contextual** `h_t` es la reescritura de `e_t` **condicionada al diálogo**:

```text
Texto del turno  u_t
   ↓ f1  (BaseTurnEncoder)
Embedding base   e_t          (depende solo de u_t)
   ↓ f2  (ContextualTurnModel, condicionado al contexto del diálogo)
Embedding contextual  h_t     (depende de u_t y de los demás turnos)
```

`h_t` busca capturar no solo *qué dice* el turno sino *qué hace* dentro de la conversación.
Dos "sí" en contextos distintos deberían recibir vectores `h_t` distintos; dos turnos con
texto distinto pero misma función conversacional deberían acercarse.

## 4. Una progresión natural de representaciones

El paquete se inscribe en una progresión de formas de representar un turno dentro de su diálogo:

| Representación | Cómo se obtiene | Aprendida | Usa contexto |
|----------------|-----------------|-----------|--------------|
| **Static / base** `e_t` | encoder de oraciones, turno aislado | no | no |
| **Cumulative (dinámica)** | promedio/acumulado de `e_1..e_t` | no | sí (causal, sin pesos aprendidos) |
| **EMA** | media móvil exponencial de `e_1..e_t` | no (salvo el coeficiente) | sí (causal, decae con la distancia) |
| **Contextual aprendida** `h_t` | `ContextualTurnModel` (este paquete) | **sí** | **sí** (atención aprendida) |

Las versiones cumulative y EMA son heurísticas **fijas**: combinan los embeddings previos con
una regla cerrada. La versión contextual **aprende** cómo combinar el contexto mediante
atención, y puede ser **bidireccional** (mira todo el diálogo) o **autoregresiva** (solo el
pasado). Ver [research_notes.md](research_notes.md) para cómo se comparan experimentalmente.

## 5. Relevancia para memoria conversacional y retrieval

En recuperación sobre diálogos (p. ej. "dado el estado actual de la conversación, recuperar
turnos/situaciones similares"), la calidad depende de que la representación distinga
**situaciones conversacionales**, no solo **superficies de texto**. Embeddings base tienden a
recuperar repeticiones léxicas (todos los "gracias" juntos). La hipótesis es que `h_t`, al
incorporar el contexto, recupera situaciones **funcionalmente** más plausibles.

## 6. La separación `f1` / `f2` (decisión de diseño)

`f1` (texto → `e_t`) y `f2` (`e_t` + contexto → `h_t`) están **desacoplados** a propósito:

- Permite usar **cualquier** encoder base (MiniLM, MPNet, o embeddings de Dialog2Flow ya
  calculados) sin reentrenarlo.
- Permite entrenar `f2` directamente sobre **embeddings precomputados**, evitando descargas y
  cómputo de `f1` en cada corrida.
- Mantiene `f2` liviano y reutilizable, y hace explícito qué parte aporta el contexto.

En v1, `f1` se trata como entrada **fija** (no se hace fine-tuning conjunto del encoder base);
es una decisión deliberada, no una limitación del diseño (ver [research_notes.md](research_notes.md)).

## 7. La idea de `H @ E.T` (anticipo)

El objetivo opcional `embedding_retrieval` lleva la analogía con los LLM un paso más allá. En
un modelo de lenguaje, la proyección final compara el estado oculto con la matriz de embeddings
del vocabulario: `h_t @ W_vocab.T → logits sobre tokens`. Acá hacemos el análogo a nivel de
turno: `h_t @ E_candidates.T → scores sobre turnos candidatos`. Es decir, tratamos los
embeddings de turnos como un "vocabulario de turnos" y pedimos que `h_t` puntúe alto al turno
objetivo correcto. El detalle completo está en [losses.md](losses.md).
