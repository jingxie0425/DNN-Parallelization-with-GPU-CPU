cnn: cnn.c
	g++ -g -fopenmp -O2 -o cnn.o -c cnn.c -I.

dense: dense.c
	g++ -g -fopenmp -O2 -o dense.o -c dense.c -I.

maxpooling: maxpool.c
	g++ -g -fopenmp -O2 -o maxpool.o -c maxpool.c -I.

read1D: read1D.cpp
	g++ -g -o read1D.o -c read1D.cpp -I.

read2D: read2D.cpp
	g++ -g -o read2D.o -c read2D.cpp -I.

relu: relu.c
	g++ -g -fopenmp -O2 -o relu.o -c relu.c -I.

softmax: softmax.c
	g++ -g -fopenmp -O2 -o softmax.o -c softmax.c -I.

writeOut: writeOut.cpp
	g++ -g -o writeOut.o -c writeOut.cpp -I.

libr: cnn.o dense.o maxpool.o read1D.o read2D.o relu.o softmax.o writeOut.o 
	ar rcs libcnn.a cnn.o dense.o maxpool.o read1D.o read2D.o relu.o softmax.o writeOut.o

all: cnn dense maxpooling read1D read2D relu softmax writeOut libr
.PHONY: clean

clean:
	rm cnn.o dense.o maxpool.o read1D.o read2D.o relu.o softmax.o writeOut.o libcnn.a

