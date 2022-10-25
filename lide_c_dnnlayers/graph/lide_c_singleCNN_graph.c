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

#include "lide_c_singleCNN_graph.h"
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
    Usage: lide_c_singleCNN_driver.exe m_file x_file y_file out_file
*/

struct _lide_c_singleCNN_graph_context_struct {
#include "lide_c_graph_context_type_common.h"
};

/*
#define OUTDIM  INPUTDIM-FILTERDIM+1
#define OUTNUM  INPUTNUM*FILTERNUM
*/

//simpe cnn model hyperparameters
//original

#define INPUTDIM1 17 //input dimension
#define INPUTNUM1 140 //input size
#define FILTERNUM1 32    //filter amount
#define FILTERNUM2 16    //filter amount
#define OUTDIM1  16    //output dimension before maxpooling
#define OUTNUM1  4480
#define OUTDIMMP 8  //output dimension after maxpooling
#define INPUTDIM2 8
#define INPUTNUM2 140 //input size
#define OUTDIM2  7    //output dimension before maxpooling
#define OUTNUM2  2240
#define PICN1   1
#define PICN2   32

#define NODENUM1 32 //node amount of first dense layer
#define NODENUM2 2 //node amount of second dense layer

//pruned
/*
#define INPUTDIM1 17 //input dimension
#define INPUTNUM1 140 //input size

#define FILTERNUM1 2    //filter amount
#define FILTERNUM2 14    //filter amount
#define OUTDIM1  16    //output dimension before maxpooling
#define OUTNUM1  280
#define OUTDIMMP 8  //output dimension after maxpooling
#define INPUTDIM2 8
#define INPUTNUM2 140 //input size
#define OUTDIM2  7    //output dimension before maxpooling
#define OUTNUM2  1960
#define PICN1   1
#define PICN2   2

#define NODENUM1 2 //node amount of first dense layer
#define NODENUM2 2 //node amount of second dense layer
*/
//*****************

#define FILTERDIM 2 //filter dimension
#define MPSTEP  2   //step the maxpooling layer takes

lide_c_singleCNN_graph_context_type *lide_c_singleCNN_graph_new(const char * WC1,const char * BC1, const char * InputsFile, 
    const char * WC2,const char * BC2,
    const char *WFD, const char* BFD, 
    const char* WD2, const char* BD2,
    const char* OutFile ){
    int token_size;
    lide_c_singleCNN_graph_context_type * context = NULL;
    context = (lide_c_singleCNN_graph_context_type *)lide_c_util_malloc(sizeof(
        lide_c_singleCNN_graph_context_type));
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
    context->fifos[FIFO_IN2C1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2C1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2C1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_C12RELU] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2MP] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_MP2C2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2C2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2C2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_C22RELU] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2FD] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2FD] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2FD] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_FD2RELU] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2D2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2D2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2D2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_D22SM] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_SM2WO] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
        
    /* construct actors */
    context->actors[ACTOR_READW_CONV2D1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2C1],
        WC1,FILTERNUM1*PICN1*FILTERDIM*FILTERDIM));
    context->actors[ACTOR_READB_CONV2D1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2C1], 
        BC1,FILTERNUM1));
    context->actors[ACTOR_READIN_CONV2D1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_IN2C1],
        InputsFile,INPUTNUM1*PICN1*INPUTDIM1*INPUTDIM1));

    context->actors[ACTOR_READW_CONV2D2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2C2],
        WC2,FILTERNUM2*PICN2*FILTERDIM*FILTERDIM));
    context->actors[ACTOR_READB_CONV2D2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2C2], 
        BC2,FILTERNUM2));

    context->actors[ACTOR_READW_FLATTENDENSE] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2FD], 
        WFD,FILTERNUM2*OUTDIM2*OUTDIM2*NODENUM1));

    context->actors[ACTOR_READB_FLATTENDENSE] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2FD], 
        BFD,NODENUM1));

    context->actors[ACTOR_READW_DENSE2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2D2], 
        WD2,NODENUM1*NODENUM2));
    context->actors[ACTOR_READB_DENSE2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2D2],
        BD2,NODENUM2));

    
    context->actors[ACTOR_CONV2D1] = (lide_c_actor_context_type*)(lide_c_conv2DHead_new(context->fifos[FIFO_IN2C1], context->fifos[FIFO_W2C1],
        context->fifos[FIFO_B2C1],context->fifos[FIFO_C12RELU],
        INPUTDIM1, INPUTNUM1, FILTERDIM, FILTERNUM1, OUTDIM1, OUTNUM1, PICN1));


    context->actors[ACTOR_CONV2D2] = (lide_c_actor_context_type*)(
        lide_c_conv2d_new(
        context->fifos[FIFO_MP2C2],
        context->fifos[FIFO_W2C2],
        context->fifos[FIFO_B2C2],
        context->fifos[FIFO_C22RELU],
        INPUTDIM2, INPUTNUM2, FILTERDIM, FILTERNUM2, OUTDIM2, OUTNUM2, PICN2));


    context->actors[ACTOR_MAXPOOL] = (lide_c_actor_context_type*)(lide_c_maxpool_new(context->fifos[FIFO_RELU2MP],context->fifos[FIFO_MP2C2],
        MPSTEP, OUTNUM1,OUTDIM1));


    context->actors[ACTOR_FLATTENDENSE] = (lide_c_actor_context_type*)(
        lide_c_flattenDense_new(
        context->fifos[FIFO_RELU2FD],
        context->fifos[FIFO_W2FD],
        context->fifos[FIFO_B2FD],
        context->fifos[FIFO_FD2RELU],
        INPUTNUM1,
        FILTERNUM2*OUTDIM2*OUTDIM2,
        NODENUM1,OUTDIM2, FILTERNUM2));



    context->actors[ACTOR_DENSE2] = (lide_c_actor_context_type*)(
        lide_c_dense_new(context->fifos[FIFO_RELU2D2],
        context->fifos[FIFO_W2D2],
        context->fifos[FIFO_B2D2],
        context->fifos[FIFO_D22SM],
        INPUTNUM1,NODENUM1, NODENUM2));


    context->actors[ACTOR_RELU1] = (lide_c_actor_context_type*)(lide_c_reluDense_new(context->fifos[FIFO_FD2RELU],
    context->fifos[FIFO_RELU2D2],INPUTNUM1,NODENUM1));

    context->actors[ACTOR_RELUCNN1] = (lide_c_actor_context_type*)(lide_c_reluCnn_new(context->fifos[FIFO_C12RELU],
    context->fifos[FIFO_RELU2MP],INPUTNUM1,OUTDIM1,FILTERNUM1));
    
    context->actors[ACTOR_RELUCNN2] = (lide_c_actor_context_type*)(lide_c_reluCnn_new(context->fifos[FIFO_C22RELU],
    context->fifos[FIFO_RELU2FD],INPUTNUM1,OUTDIM2,FILTERNUM2));


    context->actors[ACTOR_SOFTMAX] = (lide_c_actor_context_type*)(lide_c_softmax_new(context->fifos[FIFO_D22SM],
    context->fifos[FIFO_SM2WO],INPUTNUM1,NODENUM2));
    context->actors[ACTOR_WRITEOUT] = (lide_c_actor_context_type *)(lide_c_writeout_new(OutFile,
        context->fifos[FIFO_SM2WO],INPUTNUM1,NODENUM2));
   
    /* set scheduler for graph */
    context->scheduler = (lide_c_graph_scheduler_ftype)
        lide_c_singleCNN_graph_scheduler;
    return context;
}

void lide_c_singleCNN_graph_scheduler(lide_c_singleCNN_graph_context_type *context){
    lide_c_util_simple_scheduler(context->actors, context->actor_count, 
        context->descriptors);
    return;
}