cnnsingle: lide_c_cnnsingle_graph.c
	g++ -g -o lide_c_cnnsingle_graph.o  -c lide_c_cnnsingle_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \

libr: lide_c_cnnsingle_graph.o
	ar rcs libgraph.a lide_c_cnnsingle_graph.o

all: cnnsingle libr

.PHONY: clean

clean:
	rm lide_c_cnnsingle_graph.o libgraph.a
