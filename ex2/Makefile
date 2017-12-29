all: readme.pdf fig

fig:
	make -C fig

readme.pdf: readme.tex biblio.bib
	latexmk -pdf readme.tex

clean:
	rm -f readme.{pdf,aux,log}

.PHONY: fig
