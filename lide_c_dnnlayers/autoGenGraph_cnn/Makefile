cnnmulti: lide_c_cnnmulti_graph.c
	g++ -g -o lide_c_cnnmulti_graph.o  -c lide_c_cnnmulti_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \

libr: lide_c_cnnmulti_graph.o
	ar rcs libgraph.a lide_c_cnnmulti_graph.o

all: cnnmulti libr

.PHONY: clean

clean:
	rm lide_c_cnnmulti_graph.o libgraph.a
