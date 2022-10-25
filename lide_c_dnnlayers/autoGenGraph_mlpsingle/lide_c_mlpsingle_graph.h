#ifndef _lide_c_mlpsingle_graph_h
#define _lide_c_mlpsingle_graph_h
#include <stdio.h>
#include <stdlib.h>
#include "lide_c_basic.h"
#include "lide_c_actor.h"
#include "lide_c_fifo.h"
#include "lide_c_graph.h"
#include "lide_c_util.h"
#define BUFFER_CAPACITY 16
#define ACTOR_READIN_DENSE_fst 0
#define FIFO_IN2fst 0
#define ACTOR_READW_DENSE_fst 1
#define FIFO_W2fst 1
#define ACTOR_READB_DENSE_fst 2
#define FIFO_B2fst 2
#define ACTOR_DENSE_fst 3
#define FIFO_fst2RELU  3
#define ACTOR_RELUDEN_fst 4
#define ACTOR_READW_DENSE_snd 5
#define FIFO_W2snd 4
#define ACTOR_READB_DENSE_snd 6
#define FIFO_B2snd 5
#define ACTOR_DENSE_snd 7
#define FIFO_RELU2snd 6
#define FIFO_snd2SOFTMAX  7
#define ACTOR_SOFTMAX 8
#define FIFO_SM2WO 8
#define ACTOR_WRITEOUT 9
#define ACTOR_COUNT 10
#define FIFO_COUNT 9
struct _lide_c_mlpsingle_graph_context_struct;
typedef struct _lide_c_mlpsingle_graph_context_struct lide_c_mlpsingle_graph_context_type;
lide_c_mlpsingle_graph_context_type *lide_c_mlpsingle_graph_new(const char * InputsFile, 
    const char* Wfst, const char* Bfst,
    const char* Wsnd, const char* Bsnd,
    const char* OutFile );
void lide_c_mlpsingle_graph_terminate(lide_c_mlpsingle_graph_context_type *graph);
void lide_c_mlpsingle_graph_scheduler(lide_c_mlpsingle_graph_context_type *graph);
#endif
