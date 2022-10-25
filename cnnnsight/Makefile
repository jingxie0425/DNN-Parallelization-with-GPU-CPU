cnn: cnn.cu
	nvcc -arch=sm_61 -rdc=true -g -o cnn.o -c cnn.cu -I.

linkcnn: cnn.o
	nvcc -arch=sm_61 -dlink -g -o cnn_link.o cnn.o -lcudadevrt -lcudart

cnndriver: cnnDriver.cu
	nvcc -arch=sm_61 -g -o cnnDriver.o -c cnnDriver.cu -I.

cnntest: cnn_layer_test.cpp
	g++ -g -o cnn_layer_test.o -c cnn_layer_test.cpp -I.

dataTrans: dataTrans.cu
	nvcc -arch=sm_61 -g -o dataTrans.o -c dataTrans.cu -I.	

dense: dense.cu
	nvcc -arch=sm_61 -g -o dense.o -c dense.cu -I.

linkdense: dense.o
	nvcc -arch=sm_61 -dlink -g -o dense_link.o dense.o -lcudadevrt -lcudart

denseDriver: denseDriver.cu
	nvcc -arch=sm_61 -g -o denseDriver.o -c denseDriver.cu -I.

maxpoolDriver: maxpoolDriver.cu
	nvcc -arch=sm_61 -g -o maxpoolDriver.o -c maxpoolDriver.cu -I.

maxpooling: maxpooling.cu
	nvcc -arch=sm_61 -g -o maxpooling.o -c maxpooling.cu -I.

linkmaxpooling: maxpooling.o
	nvcc -arch=sm_61 -dlink -g -o maxpooling_link.o maxpooling.o -lcudadevrt -lcudart

read1D: read1D.cpp
	g++ -g -o read1D.o -c read1D.cpp -I.

read2D: read2D.cpp
	g++ -g -o read2D.o -c read2D.cpp -I.

readIn: readIn.cpp
	g++ -g -o readIn.o -c readIn.cpp -I.

relu: relu.cu
	nvcc -arch=sm_61 -g -o relu.o -c relu.cu -I.

linkrelu: relu.o
	nvcc -arch=sm_61 -dlink -g -o relu_link.o relu.o -lcudadevrt -lcudart

reluDriver: reluDriver.cu
	nvcc -arch=sm_61 -g -o reluDriver.o -c reluDriver.cu -I.

softmax: softmax.cu
	nvcc -arch=sm_61 -g -o softmax.o -c softmax.cu -I.

linksoftmax: softmax.o
	nvcc -arch=sm_61 -dlink -g -o softmax_link.o softmax.o -lcudadevrt -lcudart

softmaxDriver: softmaxDriver.cu
	nvcc -arch=sm_61 -g -o softmaxDriver.o -c softmaxDriver.cu -I.

writeOut: writeOut.cu
	nvcc -arch=sm_61 -g -o writeOut.o -c writeOut.cu -I.

writeOutDriver: writeOutDriver.cpp
	g++ -g -o writeOutDriver.o -c writeOutDriver.cpp -I.

libr:  cnn_layer_test.o cnn.o cnn_link.o cnnDriver.o dataTrans.o dense.o dense_link.o denseDriver.o maxpoolDriver.o\
	maxpooling.o maxpooling_link.o read1D.o read2D.o readIn.o relu.o relu_link.o reluDriver.o softmax.o softmax_link.o\
	softmaxDriver.o writeOut.o writeOutDriver.o
	nvcc --lib cnn_layer_test.o cnn.o cnn_link.o cnnDriver.o dataTrans.o dense.o dense_link.o denseDriver.o maxpoolDriver.o\
	 maxpooling.o maxpooling_link.o read1D.o read2D.o readIn.o relu.o relu_link.o reluDriver.o softmax.o softmax_link.o\
	 softmaxDriver.o writeOut.o writeOutDriver.o --output-file libcnn.a

cnnrun:
	g++ -o cnnrun -L. -lcnn -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

all: cnn linkcnn cnndriver cnntest dataTrans dense linkdense denseDriver maxpoolDriver maxpooling linkmaxpooling\
read1D read2D readIn relu linkrelu reluDriver softmax linksoftmax softmaxDriver writeOut writeOutDriver libr cnnrun

.PHONY: clean

clean:
	rm cnn_layer_test.o cnn.o cnn_link.o cnnDriver.o dataTrans.o dense.o dense_link.o denseDriver.o maxpoolDriver.o \
	maxpooling.o maxpooling_link.o read1D.o read2D.o readIn.o relu.o relu_link.o reluDriver.o softmax.o softmax_link.o \
	softmaxDriver.o writeOut.o writeOutDriver.o libcnn.a cnnrun

