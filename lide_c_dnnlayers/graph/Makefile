graphsimplecnn: lide_c_simpleCnn_graph.c
	g++ -g -o lide_c_simpleCnn_graph.o  -c lide_c_simpleCnn_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \

graphmultiMLP: lide_c_multiMLP_graph.c
	g++ -g -o lide_c_multiMLP_graph.o  -c lide_c_multiMLP_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \

singleMLP: lide_c_singleMLP_graph.c
	g++ -g -o lide_c_singleMLP_graph.o  -c lide_c_singleMLP_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \

multiCNN: lide_c_multiCNN_graph.c
	g++ -g -o lide_c_multiCNN_graph.o  -c lide_c_multiCNN_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \

singleCNN: lide_c_singleCNN_graph.c
	g++ -g -o lide_c_singleCNN_graph.o  -c lide_c_singleCNN_graph.c \
	-I. -I../actor -I../includes -I../../cnnnsight \


libr: lide_c_simpleCnn_graph.o lide_c_singleCNN_graph.o lide_c_multiCNN_graph.o lide_c_multiMLP_graph.o
	ar rcs libgraph.a lide_c_simpleCnn_graph.o lide_c_singleCNN_graph.o lide_c_multiCNN_graph.o lide_c_singleMLP_graph.o lide_c_multiMLP_graph.o

all: graphsimplecnn singleMLP multiCNN graphmultiMLP singleCNN libr

.PHONY: clean

clean:
	rm lide_c_simpleCnn_graph.o lide_c_multiCNN_graph.o lide_c_singleCNN_graph.o lide_c_multiMLP_graph.o lide_c_singleMLP_graph.o libgraph.a

