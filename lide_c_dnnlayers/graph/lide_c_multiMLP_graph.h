#ifndef _lide_c_multiMLP_graph_h
#define _lide_c_multiMLP_graph_h

#include <stdio.h>
#include <stdlib.h>
#include "lide_c_basic.h"
#include "lide_c_actor.h"
#include "lide_c_fifo.h"
//#include "lide_c_fifo_basic.h"
#include "lide_c_graph.h"
#include "lide_c_util.h"

#define BUFFER_CAPACITY 32

/* An enumeration of the actors in this application. */
#define ACTOR_READIN_DENSE1 0
#define ACTOR_READW_DENSE1 1
#define ACTOR_READB_DENSE1 2
#define ACTOR_READW_DENSE2 3
#define ACTOR_READB_DENSE2 4
#define ACTOR_READW_DENSE3 5
#define ACTOR_READB_DENSE3 6
#define ACTOR_READW_DENSE4 7
#define ACTOR_READB_DENSE4 8

#define ACTOR_DENSE1 	9
#define ACTOR_DENSE2 	10
#define ACTOR_DENSE3 	11
#define ACTOR_DENSE4 	12

#define ACTOR_RELU1	13
#define ACTOR_RELU2	14
#define ACTOR_RELU3	15


#define ACTOR_SOFTMAX 16
#define ACTOR_WRITEOUT 17
/* The total number of actors in the application. */
#define ACTOR_COUNT 18

/* FIFOs */
#define FIFO_W2D1	0
#define FIFO_B2D1	1
#define FIFO_W2D2	2
#define FIFO_B2D2	3
#define FIFO_W2D3	4
#define FIFO_B2D3	5
#define FIFO_W2D4	6
#define FIFO_B2D4	7

#define FIFO_RELU2D2	8
#define FIFO_RELU2D3	9
#define FIFO_RELU2D4	10

#define FIFO_D12RELU	11
#define FIFO_D22RELU	12
#define FIFO_D32RELU	13

#define FIFO_D42SM	14
#define FIFO_SM2OUT	15

#define FIFO_IN2D1 16

/* total number of FIFOs in this application */
#define FIFO_COUNT	17

struct _lide_c_multiMLP_graph_context_struct;
typedef struct _lide_c_multiMLP_graph_context_struct 
    lide_c_multiMLP_graph_context_type;

lide_c_multiMLP_graph_context_type *lide_c_multiMLP_graph_new(const char * InputsFile, 
    const char* WDense1, const char* BDense1, 
    const char* WDense2, const char* BDense2,
    const char* WDense3, const char* BDense3,
    const char* WDense4, const char* BDense4, 
    const char* OutFile );

void lide_c_multiMLP_graph_terminate(lide_c_multiMLP_graph_context_type *graph);

void lide_c_multiMLP_graph_scheduler(lide_c_multiMLP_graph_context_type *graph);

#endif