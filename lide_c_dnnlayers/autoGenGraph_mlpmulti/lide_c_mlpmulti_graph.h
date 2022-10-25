#ifndef _lide_c_mlpmulti_graph_h
#define _lide_c_mlpmulti_graph_h
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
#define ACTOR_RELUDEN_snd 8
#define FIFO_snd2RELU  7
#define ACTOR_READW_DENSE_trd 9
#define FIFO_W2trd 8
#define ACTOR_READB_DENSE_trd 10
#define FIFO_B2trd 9
#define ACTOR_DENSE_trd 11
#define FIFO_RELU2trd 10
#define ACTOR_RELUDEN_trd 12
#define FIFO_trd2RELU  11
#define ACTOR_READW_DENSE_fth 13
#define FIFO_W2fth 12
#define ACTOR_READB_DENSE_fth 14
#define FIFO_B2fth 13
#define ACTOR_DENSE_fth 15
#define FIFO_RELU2fth 14
#define FIFO_fth2SOFTMAX  15
#define ACTOR_SOFTMAX 16
#define FIFO_SM2WO 16
#define ACTOR_WRITEOUT 17
#define ACTOR_COUNT 18
#define FIFO_COUNT 17
struct _lide_c_mlpmulti_graph_context_struct;
typedef struct _lide_c_mlpmulti_graph_context_struct lide_c_mlpmulti_graph_context_type;
lide_c_mlpmulti_graph_context_type *lide_c_mlpmulti_graph_new(const char * InputsFile, 
    const char* Wfst, const char* Bfst,
    const char* Wsnd, const char* Bsnd,
    const char* Wtrd, const char* Btrd,
    const char* Wfth, const char* Bfth,
    const char* OutFile );
void lide_c_mlpmulti_graph_terminate(lide_c_mlpmulti_graph_context_type *graph);
void lide_c_mlpmulti_graph_scheduler(lide_c_mlpmulti_graph_context_type *graph);
#endif
