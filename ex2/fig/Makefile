all: graphs refresh

DOTS:=$(wildcard *.dot)
GRAPHS:=$(patsubst %.dot,%-graph.pdf,${DOTS})

graphs: ${GRAPHS}

%-graph.pdf: %.dot
	dot -T pdf -o $@ $<

refresh:
	-kill -SIGHUP mupdf
	
clean:
	rm -f ${GRAPHS}
