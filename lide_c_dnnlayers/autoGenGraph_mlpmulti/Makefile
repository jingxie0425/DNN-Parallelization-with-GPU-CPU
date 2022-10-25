mlpmulti: lide_c_mlpmulti_graph.c
	g++ -g -o lide_c_mlpmulti_graph.o  -c lide_c_mlpmulti_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \

libr: lide_c_mlpmulti_graph.o
	ar rcs libgraph.a lide_c_mlpmulti_graph.o

all: mlpmulti libr

.PHONY: clean

clean:
	rm lide_c_mlpmulti_graph.o libgraph.a
