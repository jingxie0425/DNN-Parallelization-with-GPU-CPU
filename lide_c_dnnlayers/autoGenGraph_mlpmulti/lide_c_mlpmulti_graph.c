#include <stdio.h>
#include <stdlib.h>
#include "lide_c_mlpmulti_graph.h"
#include "lide_c_dense.h"
#include "lide_c_headDense.h"
#include "lide_c_read1D.h"
#include "lide_c_reluDense.h"
#include "lide_c_softmax.h"
#include "lide_c_writeOut.h"
struct _lide_c_mlpmulti_graph_context_struct {
#include "lide_c_graph_context_type_common.h"
};
#define NODENUMfst 32
#define INPUTNUMfst 2701
#define INPUTDIMfst 140
#define NODENUMsnd 32
#define INPUTNUMsnd 2701
#define INPUTDIMsnd 140
#define NODENUMtrd 32
#define INPUTNUMtrd 2701
#define INPUTDIMtrd 140
#define NODENUMfth 2
#define INPUTNUMfth 2701
#define INPUTDIMfth 140
lide_c_mlpmulti_graph_context_type *lide_c_mlpmulti_graph_new(const char * InputsFile, 
    const char* Wfst, const char* Bfst,
    const char* Wsnd, const char* Bsnd,
    const char* Wtrd, const char* Btrd,
    const char* Wfth, const char* Bfth,
    const char* OutFile ){
    int token_size;
    lide_c_mlpmulti_graph_context_type * context = NULL;
    context = (lide_c_mlpmulti_graph_context_type *)lide_c_util_malloc(sizeof(
        lide_c_mlpmulti_graph_context_type));
    context->actor_count = ACTOR_COUNT;
    context->fifo_count = FIFO_COUNT;
    context->actors = (lide_c_actor_context_type **)lide_c_util_malloc(
        context->actor_count * sizeof(lide_c_actor_context_type *));
    context->fifos = (lide_c_fifo_pointer *)lide_c_util_malloc(
        context->fifo_count * sizeof(lide_c_fifo_pointer));
    context->descriptors = (char **)lide_c_util_malloc(context->actor_count * 
        sizeof(char*));
    token_size = sizeof(float*);
    context->fifos[FIFO_IN2fst] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2fst] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2fst] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_fst2RELU ] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2snd] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2snd] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2snd] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_snd2RELU ] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2trd] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2trd] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2trd] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_trd2RELU ] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2fth] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2fth] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2fth] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_fth2SOFTMAX ] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_SM2WO] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->actors[ACTOR_READIN_DENSE_fst] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_IN2fst], InputsFile,INPUTNUMfst*INPUTDIMfst));
    context->actors[ACTOR_READW_DENSE_fst] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2fst], Wfst,INPUTDIMfst*NODENUMfst));
    context->actors[ACTOR_READB_DENSE_fst] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2fst],Bfst,NODENUMfst));
    context->actors[ACTOR_DENSE_fst] = (lide_c_actor_context_type*)(lide_c_headDense_new(context->fifos[FIFO_IN2fst],context->fifos[FIFO_W2fst],context->fifos[FIFO_B2fst],context->fifos[FIFO_fst2RELU],INPUTNUMfst,INPUTDIMfst, NODENUMfst));
    context->actors[ACTOR_RELUDEN_fst] = (lide_c_actor_context_type*)(lide_c_reluDense_new(context->fifos[FIFO_fst2RELU],context->fifos[FIFO_RELU2snd],INPUTNUMfst,NODENUMfst));
    context->actors[ACTOR_READW_DENSE_snd] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2snd], Wsnd,NODENUMfst*NODENUMsnd));
    context->actors[ACTOR_READB_DENSE_snd] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2snd],Bsnd,NODENUMsnd));
    context->actors[ACTOR_DENSE_snd] = (lide_c_actor_context_type*)(lide_c_dense_new(context->fifos[FIFO_RELU2snd],context->fifos[FIFO_W2snd],context->fifos[FIFO_B2snd],context->fifos[FIFO_snd2RELU],INPUTNUMsnd,NODENUMfst, NODENUMsnd));
    context->actors[ACTOR_RELUDEN_snd] = (lide_c_actor_context_type*)(lide_c_reluDense_new(context->fifos[FIFO_snd2RELU],context->fifos[FIFO_RELU2trd],INPUTNUMsnd,NODENUMsnd));
    context->actors[ACTOR_READW_DENSE_trd] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2trd], Wtrd,NODENUMsnd*NODENUMtrd));
    context->actors[ACTOR_READB_DENSE_trd] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2trd],Btrd,NODENUMtrd));
    context->actors[ACTOR_DENSE_trd] = (lide_c_actor_context_type*)(lide_c_dense_new(context->fifos[FIFO_RELU2trd],context->fifos[FIFO_W2trd],context->fifos[FIFO_B2trd],context->fifos[FIFO_trd2RELU],INPUTNUMtrd,NODENUMsnd, NODENUMtrd));
    context->actors[ACTOR_RELUDEN_trd] = (lide_c_actor_context_type*)(lide_c_reluDense_new(context->fifos[FIFO_trd2RELU],context->fifos[FIFO_RELU2fth],INPUTNUMtrd,NODENUMtrd));
    context->actors[ACTOR_READW_DENSE_fth] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2fth], Wfth,NODENUMtrd*NODENUMfth));
    context->actors[ACTOR_READB_DENSE_fth] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2fth],Bfth,NODENUMfth));
    context->actors[ACTOR_DENSE_fth] = (lide_c_actor_context_type*)(lide_c_dense_new(context->fifos[FIFO_RELU2fth],context->fifos[FIFO_W2fth],context->fifos[FIFO_B2fth],context->fifos[FIFO_fth2SOFTMAX],INPUTNUMfth,NODENUMtrd, NODENUMfth));
    context->actors[ACTOR_SOFTMAX] = (lide_c_actor_context_type*)(lide_c_softmax_new(context->fifos[FIFO_fth2SOFTMAX],context->fifos[FIFO_SM2WO],INPUTNUMfth,NODENUMfth));
    context->actors[ACTOR_WRITEOUT] = (lide_c_actor_context_type *)(lide_c_writeout_new(OutFile,context->fifos[FIFO_SM2WO],INPUTNUMfth,NODENUMfth));
    context->scheduler = (lide_c_graph_scheduler_ftype)
        lide_c_mlpmulti_graph_scheduler;
    return context;
}
void lide_c_mlpmulti_graph_scheduler(lide_c_mlpmulti_graph_context_type *context){
    lide_c_util_simple_scheduler(context->actors, context->actor_count, context->descriptors);
    return;
}
