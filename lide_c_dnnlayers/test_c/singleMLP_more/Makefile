driver: driver.c
	g++ -fopenmp -O2 -o driver.o  -c driver.c -I../../autoGenGraph_mlpsingle -I../../includes
exe:
	g++ -fopenmp -O2 -o driver driver.o -L../../autoGenGraph_mlpsingle -lgraph -l:lide_c_runtime.a -L../../actor -lactor -L../../../nn_c -lcnn -L../../libs -l:lide_c_runtime.a -l:lide_c_edges_basic.a
all: driver exe

.PHONY: clean

clean:
	rm driver.o driver
