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
#include "lide_c_reluCnn.h"
#include "lide_c_util.h"
#include "lide_c_graph.h"

#include "relu.h"

#include <omp.h>

#define TIME

struct _lide_c_reluCnn_context_struct {
#include "lide_c_actor_context_type_common.h"
    /* variable. */
    unsigned int inputN;
    unsigned int oD;
    unsigned int fN;


    /* hold for outside cpu variables */
    float* inputs;
    /* Input ports. */
    lide_c_fifo_pointer inputsFIFO;
    /* Output port. */
    lide_c_fifo_pointer outsFIFO;
    

};


lide_c_reluCnn_context_type *lide_c_reluCnn_new(lide_c_fifo_pointer inputsFIFO,
    lide_c_fifo_pointer outsFIFO,unsigned int inputN,unsigned int oD, unsigned int fN) {

    lide_c_reluCnn_context_type* context = NULL;

    context = (lide_c_reluCnn_context_type*)lide_c_util_malloc(sizeof(lide_c_reluCnn_context_type));
    context->invoke = 
            (lide_c_actor_invoke_function_type)lide_c_reluCnn_invoke;
    context->enable = 
            (lide_c_actor_enable_function_type)lide_c_reluCnn_enable;

    context->mode = LIDE_C_reluCnn_LOAD_INPUTS;
    context->inputsFIFO = inputsFIFO;
    context->outsFIFO = outsFIFO;
    
    context->inputN = inputN;
    context->oD = oD;
    context->fN = fN;

    return context;
}

boolean lide_c_reluCnn_enable(lide_c_reluCnn_context_type *context) {
    boolean result = FALSE;

    switch (context->mode) {
    case LIDE_C_reluCnn_LOAD_INPUTS:
        result = lide_c_fifo_population(context->inputsFIFO) >= 1;
        break;
    case LIDE_C_reluCnn_EXECUTE:
        result = TRUE;
        break;
    default:
        result = FALSE;
        break;
    }
    return result;
}

void lide_c_reluCnn_invoke(lide_c_reluCnn_context_type *context) {
#ifdef TIME
    double time_spent = 0.0;
    double begin = omp_get_wtime();
    double end = omp_get_wtime();
#endif
    //printf("reluCnn actor invoked\n");
    switch (context->mode) {
    case LIDE_C_reluCnn_LOAD_INPUTS:
#ifdef TIME
	    time_spent = 0.0;
	    begin = omp_get_wtime();
#endif
        lide_c_fifo_read(context->inputsFIFO, &context->inputs);
        
        context->mode = LIDE_C_reluCnn_EXECUTE;
#ifdef TIME
        end = omp_get_wtime();
        time_spent += (double)(end - begin) / 1;
        printf("reluCnnLoadInput takes %f ms\n", time_spent*1000);
#endif
        break;
    case LIDE_C_reluCnn_EXECUTE:
#ifdef TIME
	    time_spent = 0.0;
	    begin = omp_get_wtime();
#endif
        reluCnn(context->inputs,context->fN,context->inputN,context->oD);
        //output a float* to CUDA memory
        lide_c_fifo_write(context->outsFIFO, &context->inputs);

        context->mode = LIDE_C_reluCnn_LOAD_INPUTS;
#ifdef TIME
        end = omp_get_wtime();
        time_spent += (double)(end - begin) / 1;
        printf("reluCnnExecute takes %f ms\n", time_spent*1000);
#endif
        break;
    default:
        context->mode = LIDE_C_reluCnn_LOAD_INPUTS;
        break;
    }
}

void lide_c_reluCnn_terminate(lide_c_reluCnn_context_type *context) {
    free(context);
}