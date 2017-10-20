all: readme.pdf

readme.pdf: graph.pdf readme.tex
	pdflatex readme.tex

graph.pdf: placement.py
	python placement.py

clean:
	rm -f graph.pdf readme.pdf readme.aux readme.log
