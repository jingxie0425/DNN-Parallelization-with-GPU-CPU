edge: lide_c_fifo.c
	gcc -o lide_c_fifo.o -c lide_c_fifo.c -I.

runtime: lide_c_util.c
	gcc -o lide_c_util.o -c lide_c_util.c -I.

wrapEdge: lide_c_fifo.o
	ar rcs lide_c_edges_basic.a lide_c_fifo.o

wrapUtil: lide_c_util.o
	ar rcs lide_c_runtime.a lide_c_util.o

all: edge runtime wrapEdge wrapUtil

.PHONY: clean

clean:
	rm lide_c_edges_basic.a lide_c_fifo.o lide_c_util.o lide_c_runtime.a 
