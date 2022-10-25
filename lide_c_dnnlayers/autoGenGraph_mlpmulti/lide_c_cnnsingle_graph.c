#include <stdio.h>
#include <stdlib.h>
#include "lide_c_cnnsingle_graph.h"
#include "lide_c_conv2D.h"
#include "lide_c_conv2DHead.h"
#include "lide_c_flattenDense.h"
#include "lide_c_maxpool.h"
#include "lide_c_read2D.h"
#include "lide_c_reluCnn.h"
#include "lide_c_dense.h"
#include "lide_c_headDense.h"
#include "lide_c_read1D.h"
#include "lide_c_reluDense.h"
#include "lide_c_softmax.h"
#include "lide_c_writeOut.h"
struct _lide_c_cnnsingle_graph_context_struct {
#include "lide_c_graph_context_type_common.h"
};
#define FILTERDIMconv1 2
#define INPUTDIMconv1 12
#define INPUTNUMconv1 2701
#define PICNconv1 1
#define FILTERNUMconv1 32
#define OUTDIMconv1 11
#define OUTNUMconv1 86432
#define MPSTEPmax_pooling2d_2 2
#define OUTDIMMPmax_pooling2d_2 5
#define FILTERNUMmax_pooling2d_2 32
#define FILTERDIMconv2 2
#define INPUTDIMconv2 5
#define INPUTNUMconv2 2701
#define PICNconv2 32
#define FILTERNUMconv2 32
#define OUTDIMconv2 4
#define OUTNUMconv2 86432
#define NODENUMdense1 32
#define INPUTNUMdense1 2701
#define INPUTDIMdense1 140
#define NODENUMdense2 2
#define INPUTNUMdense2 2701
#define INPUTDIMdense2 140
lide_c_cnnsingle_graph_context_type *lide_c_cnnsingle_graph_new(const char * InputsFile, 
    const char* Wconv1, const char* Bconv1,
    const char* Wconv2, const char* Bconv2,
    const char* Wdense1, const char* Bdense1,
    const char* Wdense2, const char* Bdense2,
    const char* OutFile ){
    int token_size;
    lide_c_cnnsingle_graph_context_type * context = NULL;
    context = (lide_c_cnnsingle_graph_context_type *)lide_c_util_malloc(sizeof(
        lide_c_cnnsingle_graph_context_type));
    context->actor_count = ACTOR_COUNT;
    context->fifo_count = FIFO_COUNT;
    context->actors = (lide_c_actor_context_type **)lide_c_util_malloc(
        context->actor_count * sizeof(lide_c_actor_context_type *));
    context->fifos = (lide_c_fifo_pointer *)lide_c_util_malloc(
        context->fifo_count * sizeof(lide_c_fifo_pointer));
    context->descriptors = (char **)lide_c_util_malloc(context->actor_count * 
        sizeof(char*));
    token_size = sizeof(float*);
    context->fifos[FIFO_W2conv1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2conv1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_IN2conv1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_conv12RELU ] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2max_pooling2d_2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2conv2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2conv2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_MP2conv2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_conv22RELU ] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2dense1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2dense1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2dense1] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_dense12RELU ] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_W2dense2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_B2dense2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_RELU2dense2] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_dense22SOFTMAX ] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->fifos[FIFO_SM2WO] = (lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);
    context->actors[ACTOR_READW_conv1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2conv1],Wconv1,FILTERNUMconv1*PICNconv1*FILTERDIMconv1*FILTERDIMconv1));
    context->actors[ACTOR_READB_conv1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2conv1], Bconv1,FILTERNUMconv1));
    context->actors[ACTOR_READIN_conv1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_IN2conv1],InputsFile,INPUTNUMconv1*PICNconv1*INPUTDIMconv1*INPUTDIMconv1));
    context->actors[ACTOR_conv1] = (lide_c_actor_context_type*)(lide_c_conv2DHead_new(context->fifos[FIFO_IN2conv1], context->fifos[FIFO_W2conv1],context->fifos[FIFO_B2conv1],context->fifos[FIFO_conv12RELU],INPUTDIMconv1, INPUTNUMconv1, FILTERDIMconv1, FILTERNUMconv1, OUTDIMconv1, OUTNUMconv1, PICNconv1));
    context->actors[ACTOR_RELUCNN_conv1] = (lide_c_actor_context_type*)(lide_c_reluCnn_new(context->fifos[FIFO_conv12RELU],context->fifos[FIFO_RELU2max_pooling2d_2],INPUTNUMconv1,OUTDIMconv1,FILTERNUMconv1));
    context->actors[ACTOR_MAXPOOL_max_pooling2d_2] = (lide_c_actor_context_type*)(lide_c_maxpool_new(context->fifos[FIFO_RELU2max_pooling2d_2],context->fifos[FIFO_MP2conv2],MPSTEPmax_pooling2d_2, OUTNUMconv1,OUTDIMconv1));
    context->actors[ACTOR_READW_conv2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2conv2],Wconv2,FILTERNUMconv2*PICNconv2*FILTERDIMconv2*FILTERDIMconv2));
    context->actors[ACTOR_READB_conv2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2conv2], Bconv2,FILTERNUMconv2));
    context->actors[ACTOR_RELUCNN_conv2] = (lide_c_actor_context_type*)(lide_c_reluCnn_new(context->fifos[FIFO_conv22RELU],context->fifos[FIFO_RELU2dense1],INPUTNUMconv2,OUTDIMconv2,FILTERNUMconv2));
    context->actors[ACTOR_conv2] = (lide_c_actor_context_type*)(lide_c_conv2d_new(context->fifos[FIFO_MP2conv2], context->fifos[FIFO_W2conv2],context->fifos[FIFO_B2conv2],context->fifos[FIFO_conv22RELU],INPUTDIMconv2, INPUTNUMconv2, FILTERDIMconv2, FILTERNUMconv2, OUTDIMconv2, OUTNUMconv2, PICNconv2));
    context->actors[ACTOR_READW_FLATTENDENSE_dense1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2dense1], Wdense1,FILTERNUMconv2*OUTDIMconv2*OUTDIMconv2*NODENUMdense1));
    context->actors[ACTOR_READB_FLATTENDENSE_dense1] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2dense1], Bdense1,NODENUMdense1));
    context->actors[ACTOR_FLATTENDENSE_dense1] = (lide_c_actor_context_type*)(lide_c_flattenDense_new(context->fifos[FIFO_RELU2dense1],context->fifos[FIFO_W2dense1],context->fifos[FIFO_B2dense1],context->fifos[FIFO_dense12RELU],INPUTNUMdense1,FILTERNUMconv2*OUTDIMconv2*OUTDIMconv2,NODENUMdense1,OUTDIMconv2, FILTERNUMconv2));
    context->actors[ACTOR_RELUDEN_dense1] = (lide_c_actor_context_type*)(lide_c_reluDense_new(context->fifos[FIFO_dense12RELU],context->fifos[FIFO_RELU2dense2],INPUTNUMdense1,NODENUMdense1));
    context->actors[ACTOR_READW_DENSE_dense2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_W2dense2], Wdense2,NODENUMdense1*NODENUMdense2));
    context->actors[ACTOR_READB_DENSE_dense2] = (lide_c_actor_context_type*)(lide_c_read1D_new(context->fifos[FIFO_B2dense2],Bdense2,NODENUMdense2));
    context->actors[ACTOR_DENSE_dense2] = (lide_c_actor_context_type*)(lide_c_dense_new(context->fifos[FIFO_RELU2dense2],context->fifos[FIFO_W2dense2],context->fifos[FIFO_B2dense2],context->fifos[FIFO_dense22SOFTMAX],INPUTNUMdense2,NODENUMdense1, NODENUMdense2));
    context->actors[ACTOR_SOFTMAX] = (lide_c_actor_context_type*)(lide_c_softmax_new(context->fifos[FIFO_dense22SOFTMAX],context->fifos[FIFO_SM2WO],INPUTNUMdense2,NODENUMdense2));
    context->actors[ACTOR_WRITEOUT] = (lide_c_actor_context_type *)(lide_c_writeout_new(OutFile,context->fifos[FIFO_SM2WO],INPUTNUMdense2,NODENUMdense2));
    context->scheduler = (lide_c_graph_scheduler_ftype)
        lide_c_cnnsingle_graph_scheduler;
    return context;
}
void lide_c_cnnsingle_graph_scheduler(lide_c_cnnsingle_graph_context_type *context){
    lide_c_util_simple_scheduler(context->actors, context->actor_count, context->descriptors);
    return;
}
