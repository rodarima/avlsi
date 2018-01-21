PDFS=report.pdf slides.pdf

all: $(PDFS)

fig/A/laplacian.pdf: plot.py
	python plot.py

%.pdf: %.tex fig/A/laplacian.pdf
	latexmk -pdf $<
	@kill -HUP $$(pidof mupdf)

clean:
	rm -f ${PDFS} *.{aux,log}
