all: readme.pdf

readme.pdf: placement.pdf vertical.pdf readme.tex floor-graph.pdf floorplan.pdf plans.pdf
	pdflatex readme.tex

placement.pdf: placement.py
	python placement.py

vertical.pdf: vertical.dot
	dot -T pdf vertical.dot -o vertical.pdf

floor-graph.pdf: floor-graph.dot
	dot -T pdf floor-graph.dot -o floor-graph.pdf

clean:
	rm -f {placement,vertical}.pdf readme.{pdf,aux,log}
