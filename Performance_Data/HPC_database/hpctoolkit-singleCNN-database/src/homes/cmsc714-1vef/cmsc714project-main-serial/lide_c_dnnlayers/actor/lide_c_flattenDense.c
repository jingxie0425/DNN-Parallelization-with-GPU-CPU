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
#include <time.h>
#include "lide_c_flattenDense.h"
#include "lide_c_util.h"
#include "lide_c_graph.h"

#include "dense.h"

#define TIME

struct _lide_c_flattenDense_context_struct {
#include "lide_c_actor_context_type_common.h"
    /* variable. */
    DENSEHANDLER densehandler;
    unsigned int inputN;
    unsigned int preNodeNum;
    unsigned int nodeNum;
    unsigned int oD;
    unsigned int fN;


    /* hold for outside cpu variables */
    float* inputs;
    float* weights;
    float* bias;

    /* Input ports. */
    lide_c_fifo_pointer inputsFIFO;
    lide_c_fifo_pointer weightsFIFO;
    lide_c_fifo_pointer biasFIFO;

    /* Output port. */
    lide_c_fifo_pointer outsFIFO;
    

};


lide_c_flattenDense_context_type *lide_c_flattenDense_new(lide_c_fifo_pointer inputsFIFO,
    lide_c_fifo_pointer weightsFIFO,lide_c_fifo_pointer biasFIFO,lide_c_fifo_pointer outsFIFO,
        unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum,
        unsigned int oD, unsigned int fN) {

    lide_c_flattenDense_context_type* context = NULL;

    context = (lide_c_flattenDense_context_type*)lide_c_util_malloc(sizeof(lide_c_flattenDense_context_type));
    context->invoke = 
            (lide_c_actor_invoke_function_type)lide_c_flattenDense_invoke;
    context->enable = 
            (lide_c_actor_enable_function_type)lide_c_flattenDense_enable;

    context->mode = LIDE_C_flattenDense_LOAD_PARAS;
    context->inputsFIFO = inputsFIFO;
    context->weightsFIFO = weightsFIFO;
    context->biasFIFO = biasFIFO;
    context->outsFIFO = outsFIFO;
    
    context->inputN = inputN;
    context->preNodeNum = preNodeNum; //FILTERNUM*oD*oD
    context->nodeNum = nodeNum;
    context->oD = oD;
    context->fN = fN;

    context->densehandler = denseNew(inputN,preNodeNum, nodeNum);

    return context;
}

boolean lide_c_flattenDense_enable(lide_c_flattenDense_context_type *context) {
    boolean result = FALSE;

    switch (context->mode) {
    case LIDE_C_flattenDense_LOAD_INPUTS:
        result = lide_c_fifo_population(context->inputsFIFO) >= 1;
        break;
    case LIDE_C_flattenDense_LOAD_PARAS:
        result = (lide_c_fifo_population(context->weightsFIFO) >= 1) && (lide_c_fifo_population(context->biasFIFO) >= 1);
        break;
    case LIDE_C_flattenDense_EXECUTE:
        result = TRUE;
        break;
    default:
        result = FALSE;
        break;
    }
    return result;
}

void lide_c_flattenDense_invoke(lide_c_flattenDense_context_type *context) {
#ifdef TIME
    double time_spent = 0.0;
    clock_t begin = clock();
    clock_t end = clock();
#endif
    switch (context->mode) {
    case LIDE_C_flattenDense_LOAD_INPUTS:
#ifdef TIME
	    time_spent = 0.0;
	    begin = clock();
#endif
        //printf("flattenDense actor load inputs\n");
        lide_c_fifo_read(context->inputsFIFO, &(context->inputs));
        
        context->mode = LIDE_C_flattenDense_EXECUTE;
#ifdef TIME
        end = clock();
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("flattenDenseLoadInput takes %f ms\n", time_spent*1000);
#endif
        break;
    case LIDE_C_flattenDense_LOAD_PARAS:
        //printf("flattenDense actor load paras\n");
        lide_c_fifo_read(context->weightsFIFO, &(context->weights));
        lide_c_fifo_read(context->biasFIFO, &(context->bias));

        context->mode = LIDE_C_flattenDense_LOAD_INPUTS;
        break;
    case LIDE_C_flattenDense_EXECUTE:
#ifdef TIME
	    time_spent = 0.0;
	    begin = clock();
#endif
        //printf("flattenDense actor execute\n");
        //load in paras
        denseLoad(context->densehandler,context->preNodeNum, context->nodeNum,context->weights,context->bias);
        //execute
        denserun(context->densehandler,context->inputN, context->nodeNum,context->inputs,context->oD,context->fN);
        //output a float* to CUDA memory
        lide_c_fifo_write(context->outsFIFO, &context->densehandler->outs);

        context->mode = LIDE_C_flattenDense_LOAD_INPUTS;
#ifdef TIME
        end = clock();
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("flattenDenseExecute takes %f ms\n", time_spent*1000);
#endif
        break;
    default:
        context->mode = LIDE_C_flattenDense_LOAD_PARAS;
        break;
    }
}

void lide_c_flattenDense_terminate(lide_c_flattenDense_context_type *context) {
    /*free cuda memory*/
    denseFree(context->densehandler);
    /*free context memory*/
    free(context);
}