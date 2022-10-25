conv2d: lide_c_conv2D.c 
	g++ -g -fopenmp -O2 -o lide_c_conv2D.o  -c lide_c_conv2D.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

conv2dhead: lide_c_conv2DHead.c 
	g++ -g -fopenmp -O2 -o lide_c_conv2DHead.o  -c lide_c_conv2DHead.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

dense: lide_c_dense.c 
	g++ -fopenmp -O2 -o lide_c_dense.o  -c lide_c_dense.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

flattenDense: lide_c_flattenDense.c 
	g++ -g -fopenmp -O2 -o lide_c_flattenDense.o  -c lide_c_flattenDense.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

headDense: lide_c_headDense.c
	g++ -g -fopenmp -O2 -o lide_c_headDense.o  -c lide_c_headDense.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

maxpool: lide_c_maxpool.c 
	g++ -fopenmp -O2 -o lide_c_maxpool.o  -c lide_c_maxpool.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

read1D: lide_c_read1D.c 
	g++ -g -o lide_c_read1D.o  -c lide_c_read1D.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

read2D: lide_c_read2D.c 
	g++ -o lide_c_read2D.o  -c lide_c_read2D.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

reluCnn: lide_c_reluCnn.c 
	g++ -fopenmp -O2 -o lide_c_reluCnn.o  -c lide_c_reluCnn.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

reluDense: lide_c_reluDense.c
	g++ -fopenmp -O2 -o lide_c_reluDense.o  -c lide_c_reluDense.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn	

softmax: lide_c_softmax.c
	g++ -fopenmp -O2 -o lide_c_softmax.o  -c lide_c_softmax.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn	

writeOut: lide_c_writeOut.c
	g++ -o lide_c_writeOut.o  -c lide_c_writeOut.c -I. -I../includes\
	 -I../../cnnnsight -L../../cnnnsight -lcnn

libr:  lide_c_conv2D.o lide_c_conv2DHead.o lide_c_dense.o lide_c_headDense.o lide_c_flattenDense.o \
	lide_c_maxpool.o lide_c_read1D.o lide_c_read2D.o lide_c_reluCnn.o \
	lide_c_reluDense.o lide_c_softmax.o lide_c_writeOut.o
	ar rcs libactor.a lide_c_conv2D.o lide_c_conv2DHead.o lide_c_dense.o lide_c_headDense.o lide_c_flattenDense.o \
	lide_c_maxpool.o lide_c_read1D.o lide_c_read2D.o lide_c_reluCnn.o \
	lide_c_reluDense.o lide_c_softmax.o lide_c_writeOut.o


all: conv2d conv2dhead dense headDense flattenDense maxpool read1D read2D reluCnn reluDense softmax writeOut libr

.PHONY: clean

clean:
	rm lide_c_conv2D.o lide_c_dense.o lide_c_headDense.o lide_c_flattenDense.o \
	lide_c_maxpool.o lide_c_read1D.o lide_c_read2D.o lide_c_reluCnn.o \
	lide_c_reluDense.o lide_c_softmax.o lide_c_writeOut.o lide_c_conv2DHead.o libactor.a