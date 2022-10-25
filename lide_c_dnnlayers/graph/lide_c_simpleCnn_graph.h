#ifndef _lide_c_simpleCnn_graph_h
#define _lide_c_simpleCnn_graph_h

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
#define ACTOR_READW_CONV2D 0
#define ACTOR_READB_CONV2D 1
#define ACTOR_READIN_CONV2D 2
#define ACTOR_CONV2D 3
#define ACTOR_MAXPOOL 4
#define ACTOR_READW_FLATTENDENSE	5
#define ACTOR_READB_FLATTENDENSE	6
#define ACTOR_FLATTENDENSE 7
#define ACTOR_READW_DENSE  8
#define ACTOR_READB_DENSE  9
#define ACTOR_DENSE 10
#define ACTOR_RELU 11
#define ACTOR_RELUCONV 12
#define ACTOR_SOFTMAX 13
#define ACTOR_WRITEOUT 14
/* The total number of actors in the application. */
#define ACTOR_COUNT 15

/* FIFOs */
#define FIFO_W2CONV	0
#define FIFO_B2CONV	1
#define FIFO_IN2CONV	2
#define FIFO_CONV2RELU	3
#define FIFO_RELU2MP	4
#define FIFO_MP2FD	5
#define FIFO_FD2RELU	6
#define FIFO_RELU2D	7
#define FIFO_D2SM	8
#define FIFO_SM2OUT	9
#define FIFO_W2FD	10
#define FIFO_B2FD	11
#define FIFO_W2D	12
#define FIFO_B2D	13
/* total number of FIFOs in this application */
#define FIFO_COUNT	14

struct _lide_c_simpleCnn_graph_context_struct;
typedef struct _lide_c_simpleCnn_graph_context_struct 
    lide_c_simpleCnn_graph_context_type;

lide_c_simpleCnn_graph_context_type *lide_c_simpleCnn_graph_new(const char * WConvFile,const char * BConvFile, const char * InputsFile, 
    const char *WFlattenDenseFile, const char* BFlattenDenseFile, const char* WDenseFile, const char* BDenseFile, const char* OutFile );

void lide_c_simpleCnn_graph_terminate(lide_c_simpleCnn_graph_context_type *graph);

void lide_c_simpleCnn_graph_scheduler(lide_c_simpleCnn_graph_context_type *graph);

#endif