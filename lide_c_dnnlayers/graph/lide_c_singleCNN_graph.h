#ifndef _lide_c_singleCNN_graph_h
#define _lide_c_singleCNN_graph_h

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
#define ACTOR_READW_CONV2D1 0
#define ACTOR_READB_CONV2D1 1
#define ACTOR_READIN_CONV2D1 2
#define ACTOR_READW_CONV2D2 3
#define ACTOR_READB_CONV2D2 4
#define ACTOR_READW_FLATTENDENSE	5
#define ACTOR_READB_FLATTENDENSE	6

#define ACTOR_READW_DENSE2  7
#define ACTOR_READB_DENSE2  8

#define ACTOR_CONV2D1 9
#define ACTOR_CONV2D2 10
#define ACTOR_MAXPOOL 11

#define ACTOR_FLATTENDENSE 12
#define ACTOR_DENSE2 13

#define ACTOR_RELUCNN1 14
#define ACTOR_RELUCNN2 15
#define ACTOR_RELU1 16

#define ACTOR_SOFTMAX 17
#define ACTOR_WRITEOUT 18
/* The total number of actors in the application. */
#define ACTOR_COUNT 19

/* FIFOs */
#define FIFO_IN2C1	0
#define FIFO_B2C1	1
#define FIFO_W2C1	2
#define FIFO_C12RELU	3
#define FIFO_RELU2MP	4
#define FIFO_MP2C2	5
#define FIFO_B2C2	6
#define FIFO_W2C2	7
#define FIFO_C22RELU	8
#define FIFO_RELU2FD	9
#define FIFO_B2FD	10
#define FIFO_W2FD	11
#define FIFO_FD2RELU	12
#define FIFO_RELU2D2	13
#define FIFO_B2D2	14
#define FIFO_W2D2	15
#define FIFO_D22SM	16
#define FIFO_SM2WO	17

/* total number of FIFOs in this application */
#define FIFO_COUNT	18

struct _lide_c_singleCNN_graph_context_struct;
typedef struct _lide_c_singleCNN_graph_context_struct 
    lide_c_singleCNN_graph_context_type;

lide_c_singleCNN_graph_context_type *lide_c_singleCNN_graph_new(
	const char * WC1,const char * BC1, 
	const char * InputsFile, 
    const char * WC2,const char * BC2,
    const char *WFD, const char* BFD, 
    const char* WD2, const char* BD2,
    const char* OutFile );

void lide_c_singleCNN_graph_terminate(lide_c_singleCNN_graph_context_type *graph);

void lide_c_singleCNN_graph_scheduler(lide_c_singleCNN_graph_context_type *graph);

#endif