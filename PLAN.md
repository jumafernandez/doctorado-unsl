# Plan de tesis doctoral

> Resumen versionado del plan de trabajo. Documento de trabajo (versión viva):
> [Google Docs](https://docs.google.com/document/d/10kLlYUa0xdmvAgl3k0UBwHb0-oEDgZNddLcM7jNs7wA/edit?usp=sharing).

## Título

*Integración de modelos de lenguaje y planificación a partir de flujos conversacionales inducidos
en agentes híbridos para sistemas de diálogo orientados a tareas.*

## Doctorando y dirección

- **Doctorando:** Mg. Juan Manuel Fernández.
- **Directores:** Dr. Sergio Burdisso (Idiap Research Institute, Suiza) y Dr. Marcelo Errecalde
  (LIDIC, Universidad Nacional de San Luis).
- **Marco institucional:** trabajo desarrollado en el **LICDIA** (Laboratorio de Ciencia de Datos
  e IA, Universidad Nacional de Luján), con colaboración de **LIDIC** (UNSL) e **Idiap** (Suiza).

## Objetivo general

Explorar y diseñar un enfoque **híbrido (neuro-simbólico)** para sistemas de diálogo orientados a
tareas (TOD) que combine **modelos de lenguaje** con **estructuras simbólicas inducidas a partir de
conversaciones reales**, con el fin de construir agentes más **consistentes, explicables y
robustos**.

## Objetivos específicos

1. **Extraer representaciones simbólicas** de datos conversacionales *no anotados*: identificar
   regiones de intención y construir grafos de transición que describan la dinámica del diálogo.
2. **Formalizar los grafos como procesos de decisión (MDP)**: definir estados, acciones,
   transiciones y recompensas.
3. **Integrar modelos de lenguaje bajo control simbólico/neuro-simbólico**, restringiendo las
   acciones del agente a los grafos inducidos.
4. **Diseñar métricas de similitud entre diálogos** basadas en los grafos de transición, para
   evaluar la coherencia estructural.

## Ejes temáticos

La investigación se ubica en la intersección de aprendizaje **simbólico**, **subsimbólico** y
**neuro-simbólico**, abarcando:

- técnicas de aprendizaje simbólico y subsimbólico;
- integración neuro-simbólica con grafos de diálogo;
- modelado de diálogo orientado a tareas (TOD);
- métodos y métricas de evaluación;
- aplicaciones de interacción humano-computadora.

## Trabajo experimental

Enfoque en cuatro etapas:

1. derivar representaciones simbólicas mediante inducción automática de estructura;
2. formalizar los grafos inducidos como MDP;
3. desarrollar esquemas híbridos con generación de LLM bajo control simbólico;
4. evaluar mediante métricas de similitud y análisis comparativo.

## Cronograma

Plan a **seis semestres**, desde la revisión del estado del arte hasta la escritura de la tesis y
la publicación de las contribuciones.

## Relación con este repositorio

- [`contextual-turn-embeddings/`](contextual-turn-embeddings/README.md) — pieza de código que aporta
  a la **base representacional** de la línea: aprende representaciones contextuales de turnos sobre
  datos conversacionales (estilo Dialog2Flow), insumo para **inducir estructura** (objetivo 1) y
  para las **métricas de similitud entre diálogos** (objetivo 4).
- [`doctorado-escrito/`](doctorado-escrito/README.md) — fuentes LaTeX de la tesis (marco teórico,
  metodología y resultados).
