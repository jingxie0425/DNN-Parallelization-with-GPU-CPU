/*******************************************************************************
@ddblock_begin copyright

Copyright (c) 1997-2018
Maryland DSPCAD Research Group, The University of Maryland at College Park 

Permission is hereby granted, without written agreement and without
license or royalty fees, to use, copy, modify, and distribute this
software and its documentation for any purpose, provided that the above
copyright notice and the following two paragraphs appear in all copies
of this software.

IN NO EVENT SHALL THE UNIVERSITY OF MARYLAND BE LIABLE TO ANY PARTY
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
THE UNIVERSITY OF MARYLAND HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

THE UNIVERSITY OF MARYLAND SPECIFICALLY DISCLAIMS ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
MARYLAND HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.

@ddblock_end copyright
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "lide_c_singleMLP_graph.h"

#include "lide_c_dense.h"
#include "lide_c_headDense.h"


#include "lide_c_read1D.h"
#include "lide_c_read2D.h"

#include "lide_c_reluDense.h"
#include "lide_c_softmax.h"
#include "lide_c_writeOut.h"

/* 
    Usage: lide_c_singleMLP_driver.exe m_file x_file y_file out_file
*/

struct _lide_c_singleMLP_graph_context_struct {
#include "lide_c_graph_context_type_common.h"
};

//simpe cnn model hyperparameters
#define INPUTDIM 273 //input dimension
#define INPUTNUM 140 //input size

/*
//pruned
#define NODENUM1 2 //node amount of first dense layer
#define NODENUM2 2
*/

//original
#define NODENUM1 32 //node amount of first dense layer
#define NODENUM2 2


lide_c_singleMLP_graph_context_type *lide_c_singleMLP_graph_new(const char * InputsFile, 
    const char *WDense1, const char* BDense1, 
    const char* WDense2, const char* BDense2,
    const char* OutFile ){

    int token_size;
    lide_c_singleMLP_graph_context_type * context = NULL;
    context = (lide_c_singleMLP_graph_context_type *)lide_c_util_malloc(sizeof(
        lide_c_singleMLP_graph_context_type));
    context->actor_count = ACTOR_COUNT;
    context->fifo_count = FIFO_COUNT;

    context->actors = (lide_c_actor_context_type **)lide_c_util_malloc(
        context->actor_count * sizeof(lide_c_actor_context_type *));
    context->fifos = (lide_c_fifo_pointer *)lide_c_util_malloc(
        context->fifo_count * sizeof(lide_c_fifo_pointer));

    context->descriptors = (char **)lide_c_util_malloc(context->actor_count * 
        sizeof(char*));

    /* construct FIFOs */
    token_size = sizeof(float*);
    context->fifos[FIFO_W2D1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2D1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_IN2D1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_D12RELU] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2D2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2D2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2D2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_D22SM] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_SM2OUT] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
        
    /* construct actors */
    context->actors[ACTOR_READIN_DENSE1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_IN2D1], 
        InputsFile,INPUTNUM*INPUTDIM));
    context->actors[ACTOR_READW_DENSE1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2D1], 
        WDense1,INPUTDIM*NODENUM1));
    context->actors[ACTOR_READB_DENSE1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2D1],
        BDense1,NODENUM1));
    context->actors[ACTOR_READW_DENSE2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2D2], 
        WDense2,NODENUM1*NODENUM2));
    context->actors[ACTOR_READB_DENSE2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2D2],
        BDense2,NODENUM2));


    context->actors[ACTOR_DENSE1] = (lide_c_actor_context_type*)(lide_c_headDense_new(context->fifos[FIFO_IN2D1],
    context->fifos[FIFO_W2D1],context->fifos[FIFO_B2D1],context->fifos[FIFO_D12RELU],INPUTNUM,INPUTDIM, NODENUM1));

    context->actors[ACTOR_DENSE2] = (lide_c_actor_context_type*)(lide_c_dense_new(context->fifos[FIFO_RELU2D2],
    context->fifos[FIFO_W2D2],context->fifos[FIFO_B2D2],context->fifos[FIFO_D22SM],INPUTNUM,NODENUM1, NODENUM2));


    context->actors[ACTOR_RELU1] = (lide_c_actor_context_type*)(lide_c_reluDense_new(context->fifos[FIFO_D12RELU],
    context->fifos[FIFO_RELU2D2],INPUTNUM,NODENUM1));

    context->actors[ACTOR_SOFTMAX] = (lide_c_actor_context_type*)(lide_c_softmax_new(context->fifos[FIFO_D22SM],
    context->fifos[FIFO_SM2OUT],INPUTNUM,NODENUM2));
    
    context->actors[ACTOR_WRITEOUT] = (lide_c_actor_context_type *)(lide_c_writeout_new(OutFile,
        context->fifos[FIFO_SM2OUT],INPUTNUM,NODENUM2));
   
    /* set scheduler for graph */
    context->scheduler = (lide_c_graph_scheduler_ftype)
        lide_c_singleMLP_graph_scheduler;
    return context;
}

void lide_c_singleMLP_graph_scheduler(lide_c_singleMLP_graph_context_type *context){
    lide_c_util_simple_scheduler(context->actors, context->actor_count, 
        context->descriptors);
    return;
}