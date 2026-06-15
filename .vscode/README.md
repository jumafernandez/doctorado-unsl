# `.vscode/` — configuración versionada a propósito

Este `.vscode/` **se versiona intencionalmente**; no es desprolijidad ni configuración
personal de una máquina.

Este repositorio es un *monorepo* del doctorado que contiene dos cosas:

- `contextual-turn-embeddings/` — el paquete de embeddings contextuales de turnos.
- `doctorado-escrito/` — la tesis en LaTeX.

`settings.json` configura **LaTeX Workshop** para compilar la tesis con `latexmk`
(que lee `doctorado-escrito/.latexmkrc`: salida a `build/` y `-shell-escape` para `minted`).
Se comparte en el repo porque:

- es **portable** — usa `%DIR%`/rutas relativas, sin rutas absolutas de ninguna máquina;
- garantiza que la **compilación de la tesis funcione igual en cualquier clon**, sin tener
  que reconfigurar el editor a mano.

Si en algún momento se prefiere no versionarlo, basta con dejar de trackearlo
(`git rm --cached .vscode/settings.json`) y agregar `.vscode/` al `.gitignore` de la raíz.
