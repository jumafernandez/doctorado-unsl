# Configuración de compilación para latexmk
# Uso:  latexmk            (compila thesis.tex y deja el PDF en build/)
#       latexmk -c         (limpia archivos auxiliares)

$pdf_mode = 1;                     # generar PDF con pdflatex
$out_dir  = 'build';               # toda la salida va a build/
$bibtex_use = 2;                   # correr bibtex automáticamente si hace falta

# -shell-escape es necesario para minted (resaltado de código con Pygments)
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 -file-line-error %O %S';

@default_files = ('thesis.tex');
