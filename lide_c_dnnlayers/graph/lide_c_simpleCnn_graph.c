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

#include "lide_c_simpleCnn_graph.h"
#include "lide_c_conv2D.h"
#include "lide_c_conv2DHead.h"
#include "lide_c_dense.h"
#include "lide_c_flattenDense.h"
#include "lide_c_maxpool.h"
#include "lide_c_read1D.h"
#include "lide_c_read2D.h"
#include "lide_c_reluCnn.h"
#include "lide_c_reluDense.h"
#include "lide_c_softmax.h"
#include "lide_c_writeOut.h"

/* 
    Usage: lide_c_simpleCnn_driver.exe m_file x_file y_file out_file
*/

struct _lide_c_simpleCnn_graph_context_struct {
#include "lide_c_graph_context_type_common.h"
};

//simpe cnn model hyperparameters
#define INPUTDIM 17 //input dimension
#define INPUTNUM 78 //input size
#define FILTERDIM 2 //filter dimension
#define FILTERNUM 32    //filter amount
#define NODENUM 16 //node amount of first dense layer
#define NODENUM2 2 //node amount of second dense layer
#define OUTDIM  16    //output dimension before maxpooling
#define OUTNUM  2496      //output amount
#define MPSTEP  2   //step the maxpooling layer takes
#define OUTDIMMP 8  //output dimension after maxpooling
#define PICN 1

lide_c_simpleCnn_graph_context_type *lide_c_simpleCnn_graph_new(const char * WConvFile,const char * BConvFile, const char * InputsFile, 
    const char *WFlattenDenseFile, const char* BFlattenDenseFile, const char* WDenseFile, const char* BDenseFile, const char* OutFile ){
    int token_size;
    lide_c_simpleCnn_graph_context_type * context = NULL;
    context = (lide_c_simpleCnn_graph_context_type *)lide_c_util_malloc(sizeof(
        lide_c_simpleCnn_graph_context_type));
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
    context->fifos[FIFO_W2CONV] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2CONV] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_IN2CONV] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_CONV2RELU] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2MP] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_MP2FD] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_FD2RELU] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2D] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_D2SM] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_SM2OUT] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2FD] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2FD] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2D] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2D] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
        
    /* construct actors */
    context->actors[ACTOR_READW_CONV2D] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2CONV],
        WConvFile,FILTERNUM*FILTERDIM*FILTERDIM));
    context->actors[ACTOR_READB_CONV2D] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2CONV], 
        BConvFile,FILTERNUM));
    context->actors[ACTOR_READIN_CONV2D] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_IN2CONV],
        InputsFile,INPUTNUM*INPUTDIM*INPUTDIM));
    context->actors[ACTOR_READW_DENSE] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2D], 
        WDenseFile,NODENUM*NODENUM2));
    context->actors[ACTOR_READB_DENSE] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2D],
        BDenseFile,NODENUM2));
    context->actors[ACTOR_READW_FLATTENDENSE] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2FD], 
        WFlattenDenseFile,FILTERNUM*OUTDIMMP*OUTDIMMP*NODENUM));
    context->actors[ACTOR_READB_FLATTENDENSE] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2FD], 
        BFlattenDenseFile,NODENUM));
    
    context->actors[ACTOR_CONV2D] = (lide_c_actor_context_type*)(lide_c_conv2DHead_new(context->fifos[FIFO_IN2CONV], context->fifos[FIFO_W2CONV],
        context->fifos[FIFO_B2CONV],context->fifos[FIFO_CONV2RELU],
        INPUTDIM, INPUTNUM, FILTERDIM, FILTERNUM, OUTDIM, OUTNUM,PICN));
    context->actors[ACTOR_MAXPOOL] = (lide_c_actor_context_type*)(lide_c_maxpool_new(context->fifos[FIFO_RELU2MP],context->fifos[FIFO_MP2FD],
        MPSTEP, OUTNUM,OUTDIM));
       
    context->actors[ACTOR_FLATTENDENSE] = (lide_c_actor_context_type*)(lide_c_flattenDense_new(context->fifos[FIFO_MP2FD],
    context->fifos[FIFO_W2FD],context->fifos[FIFO_B2FD],context->fifos[FIFO_FD2RELU],
        INPUTNUM,FILTERNUM*OUTDIMMP*OUTDIMMP,NODENUM,OUTDIMMP, FILTERNUM));

    context->actors[ACTOR_DENSE] = (lide_c_actor_context_type*)(lide_c_dense_new(context->fifos[FIFO_RELU2D],
    context->fifos[FIFO_W2D],context->fifos[FIFO_B2D],context->fifos[FIFO_D2SM],
        INPUTNUM,NODENUM, NODENUM2));

    context->actors[ACTOR_RELU] = (lide_c_actor_context_type*)(lide_c_reluDense_new(context->fifos[FIFO_FD2RELU],
    context->fifos[FIFO_RELU2D],INPUTNUM,NODENUM));
    context->actors[ACTOR_RELUCONV] = (lide_c_actor_context_type*)(lide_c_reluCnn_new(context->fifos[FIFO_CONV2RELU],
    context->fifos[FIFO_RELU2MP],INPUTNUM,OUTDIM,FILTERNUM));
    context->actors[ACTOR_SOFTMAX] = (lide_c_actor_context_type*)(lide_c_softmax_new(context->fifos[FIFO_D2SM],
    context->fifos[FIFO_SM2OUT],INPUTNUM,NODENUM2));
    context->actors[ACTOR_WRITEOUT] = (lide_c_actor_context_type *)(lide_c_writeout_new(OutFile,
        context->fifos[FIFO_SM2OUT],INPUTNUM,NODENUM2));
   
    /* set scheduler for graph */
    context->scheduler = (lide_c_graph_scheduler_ftype)
        lide_c_simpleCnn_graph_scheduler;
    return context;
}

void lide_c_simpleCnn_graph_scheduler(lide_c_simpleCnn_graph_context_type *context){
    lide_c_util_simple_scheduler(context->actors, context->actor_count, 
        context->descriptors);
    return;
}