all: readme.pdf

readme.pdf: placement.pdf vertical.pdf readme.tex
	pdflatex readme.tex

placement.pdf: placement.py
	python placement.py

vertical.pdf: vertical.dot
	dot -T pdf vertical.dot -o vertical.pdf

clean:
	rm -f graph.pdf readme.pdf readme.aux readme.log
