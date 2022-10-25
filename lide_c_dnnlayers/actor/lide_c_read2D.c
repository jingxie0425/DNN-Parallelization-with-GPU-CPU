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

#include "lide_c_read2D.h"
#include "lide_c_util.h"
#include "lide_c_graph.h"

#include "read2D.h"

struct _lide_c_read2D_context_struct {
#include "lide_c_actor_context_type_common.h"
    /* variable. */
    Read2D* read2Dptr;
    float* dataptr;
    /* Output port. */
    lide_c_fifo_pointer outsFIFO;
};


lide_c_read2D_context_type *lide_c_read2D_new(lide_c_fifo_pointer outsFIFO,
    const char* folderIn, unsigned int PicNumIn1, 
    unsigned int PicNumIn2, unsigned int PicSizeIn){

    lide_c_read2D_context_type *context = NULL;

    context = (lide_c_read2D_context_type*)lide_c_util_malloc(sizeof(lide_c_read2D_context_type));
    context->invoke = 
            (lide_c_actor_invoke_function_type)lide_c_read2D_invoke;
    context->enable = 
            (lide_c_actor_enable_function_type)lide_c_read2D_enable;

    context->mode = LIDE_C_read2D_EXECUTE;
    context->outsFIFO = outsFIFO;
    
    context->read2Dptr = new Read2D(folderIn, PicNumIn1,PicNumIn2,PicSizeIn);

    return context;
}

boolean lide_c_read2D_enable(lide_c_read2D_context_type *context) {
    boolean result = FALSE;

    switch (context->mode) {
    case LIDE_C_read2D_INACTIVE:
        result = FALSE;
        break;
    case LIDE_C_read2D_EXECUTE:
        result = TRUE;
        break;
    default:
        result = FALSE;
        break;
    }
    return result;
}

void lide_c_read2D_invoke(lide_c_read2D_context_type *context) {
    //printf("read2D actor invoked\n");
    switch (context->mode) {
    case LIDE_C_read2D_INACTIVE:
        context->mode = LIDE_C_read2D_INACTIVE;
        break;
    case LIDE_C_read2D_EXECUTE:
        context->read2Dptr->readExecute();
        //output a float* to cpu memory
        context->dataptr = context->read2Dptr->getData();
        lide_c_fifo_write(context->outsFIFO,&context->dataptr);

        context->mode = LIDE_C_read2D_INACTIVE;
        break;
    default:
        context->mode = LIDE_C_read2D_INACTIVE;
        break;
    }
}

void lide_c_read2D_terminate(lide_c_read2D_context_type *context) {
    //TODO: implement deconstructor 
    free(context);
}