all: slides.pdf

slides.pdf: slides.tex
	latexmk -pdf slides.tex
	@kill -HUP $$(pidof mupdf)
clean:
	rm -f {pdf,aux,log}
