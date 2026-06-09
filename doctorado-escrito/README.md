# Tesis doctoral — UNSL

Fuentes LaTeX de la tesis de doctorado (Ciencias de la Computación, Universidad
Nacional de San Luis). Estructura modular y limpia, lista para empezar a escribir.

## Estructura

```
doctorado-escrito/
├── thesis.tex            # documento raíz (orquesta todo)
├── references.bib        # bibliografía (BibTeX)
├── abbr.tex              # lista de abreviaturas / acrónimos
├── .latexmkrc            # configuración de compilación
│
├── config/
│   ├── packages.tex      # paquetes y configuración (preámbulo)
│   ├── commands.tex      # colores, minted, teoremas, algoritmos, macros
│   └── metadata.tex      # título, autor, director, año  ← EDITAR ACÁ
│
├── styles/
│   ├── rac.sty           # formato de tesis exigido (UNSL/Rackham) — se usa
│   ├── aas_macros.sty    # macros astronómicas (heredado, NO se carga)
│   └── agu04.bst         # estilo bibliográfico AGU (alternativo)
│
├── chapters/             # 00_resumen ... 08_conclusiones
├── appendices/           # A, B, ...
├── images/<capítulo>/    # figuras organizadas por capítulo
├── tables/<capítulo>/    # tablas organizadas por capítulo
└── build/                # salida de la compilación (no se versiona)
```

## Cómo compilar

**VS Code + LaTeX Workshop** (setup recomendado): abrí el repo `doctorado-unsl`
y apretá ▶ (o guardá, compila al guardar). La configuración está en
`.vscode/settings.json` (raíz del repo) y en `doctorado-escrito/.latexmkrc`:
el PDF queda en `build/` y `minted` funciona solo (shell-escape ya activado).

**Terminal** (necesita TeX Live y `latexmk`):

```bash
cd doctorado-escrito
latexmk          # genera build/thesis.pdf
latexmk -c       # limpia auxiliares
```

**Overleaf:** subir la carpeta y compilar con pdfLaTeX (minted funciona sin
configuración extra).

### Requisitos
- TeX Live **completo** (o Overleaf).
- Para `minted`: **Pygments** (`pip install Pygments`) y compilar con
  `-shell-escape` (ya configurado en `.latexmkrc`). Si preferís no depender de
  esto, comentá `minted` en `config/packages.tex` y usá `listings`.

## Editar los datos de la portada

Todo en `config/metadata.tex`: título, autor, director, co-director, año.

## Qué se modernizó respecto del template original

- Se eliminó el conflicto **`color` vs `xcolor`** (ahora solo `xcolor`), que era
  la causa de los errores de compilación.
- Se quitaron paquetes **duplicados** (`graphicx`, `amsmath`, `setspace`...) y
  otros innecesarios (`epsfig`, `CJKutf8`, `fontenc` griego).
- Se agregaron `microtype` y `booktabs`; `hyperref` se carga al final y se sumó
  `cleveref` (referencias en español).
- Entornos de **teoremas** y **algoritmos** traducidos al español.
- Preámbulo separado en `config/` y estructura por carpetas.
- En `rac.sty`: el **año** de la portada dejó de estar fijo en 2020 (ahora sale
  de `metadata.tex`) y se corrigieron encabezados en inglés.

## Recuperar contenido del template original

La versión anterior (tesis de S. Burdisso, con todos sus capítulos, imágenes y
las ~1669 referencias) sigue en el historial de git:

```bash
git log --oneline                                   # ver commits
git show d795f9b:doctorado_escrito/references.bib    # ver un archivo viejo
git show d795f9b:doctorado_escrito/chapters/cap1.tex
```

## Nota: monorepo

Este directorio (`doctorado-escrito`) es la **tesis escrita**. El repositorio
puede alojar carpetas hermanas como `doctorado-publicaciones`, `doctorado-src`, etc.
