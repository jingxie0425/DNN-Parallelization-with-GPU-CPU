#ifndef _lide_c_singleMLP_graph_h
#define _lide_c_singleMLP_graph_h

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

#define ACTOR_DENSE1 	5
#define ACTOR_DENSE2 	6

#define ACTOR_RELU1	7

#define ACTOR_SOFTMAX 8
#define ACTOR_WRITEOUT 9
/* The total number of actors in the application. */
#define ACTOR_COUNT 10

/* FIFOs */
#define FIFO_W2D1	0
#define FIFO_B2D1	1
#define FIFO_IN2D1 2
#define FIFO_W2D2	3
#define FIFO_B2D2	4
#define FIFO_RELU2D2	5

#define FIFO_D12RELU	6

#define FIFO_D22SM	7
#define FIFO_SM2OUT	8



/* total number of FIFOs in this application */
#define FIFO_COUNT	9

struct _lide_c_singleMLP_graph_context_struct;
typedef struct _lide_c_singleMLP_graph_context_struct 
    lide_c_singleMLP_graph_context_type;

lide_c_singleMLP_graph_context_type *lide_c_singleMLP_graph_new(const char * InputsFile, 
    const char *WDense1, const char* BDense1, 
    const char* WDense2, const char* BDense2,
    const char* OutFile );

void lide_c_singleMLP_graph_terminate(lide_c_singleMLP_graph_context_type *graph);

void lide_c_singleMLP_graph_scheduler(lide_c_singleMLP_graph_context_type *graph);

#endif