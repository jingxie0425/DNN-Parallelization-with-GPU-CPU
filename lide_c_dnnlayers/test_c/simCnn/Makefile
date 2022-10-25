driver: driver.c
	g++ -o driver.o  -c driver.c -I../../graph -I../../includes
exe:
	g++ -o driver driver.o -L../../graph -lgraph -l:lide_c_runtime.a -L../../actor -lactor -L../../../nn_c -lcnn -L../../libs -l:lide_c_runtime.a -l:lide_c_edges_basic.a
all: driver exe

.PHONY: clean

clean:
	rm driver.o driver
