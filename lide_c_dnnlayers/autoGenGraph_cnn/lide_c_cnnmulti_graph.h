#ifndef _lide_c_cnnmulti_graph_h
#define _lide_c_cnnmulti_graph_h
#include <stdio.h>
#include <stdlib.h>
#include "lide_c_basic.h"
#include "lide_c_actor.h"
#include "lide_c_fifo.h"
#include "lide_c_graph.h"
#include "lide_c_util.h"
#define BUFFER_CAPACITY 16
#define ACTOR_READW_conv1 0
#define FIFO_W2conv1 0
#define ACTOR_READB_conv1 1
#define FIFO_B2conv1 1
#define ACTOR_READIN_conv1 2
#define FIFO_IN2conv1 2
#define ACTOR_conv1 3
#define FIFO_conv12RELU  3
#define ACTOR_RELUCNN_conv1 4
#define ACTOR_MAXPOOL_max_pooling2d_1 5
#define FIFO_RELU2max_pooling2d_1 4
#define ACTOR_READW_conv2 6
#define FIFO_W2conv2 5
#define ACTOR_READB_conv2 7
#define FIFO_B2conv2 6
#define ACTOR_conv2 8
#define FIFO_MP2conv2 7
#define FIFO_conv22RELU  8
#define ACTOR_RELUCNN_conv2 9
#define ACTOR_READW_FLATTENDENSE_dense1 10
#define FIFO_W2dense1 9
#define ACTOR_READB_FLATTENDENSE_dense1 11
#define FIFO_B2dense1 10
#define ACTOR_FLATTENDENSE_dense1 12
#define FIFO_RELU2dense1 11
#define ACTOR_RELUDEN_dense1 13
#define FIFO_dense12RELU  12
#define ACTOR_READW_DENSE_dense2 14
#define FIFO_W2dense2 13
#define ACTOR_READB_DENSE_dense2 15
#define FIFO_B2dense2 14
#define ACTOR_DENSE_dense2 16
#define FIFO_RELU2dense2 15
#define ACTOR_RELUDEN_dense2 17
#define FIFO_dense22RELU  16
#define ACTOR_READW_DENSE_dense3 18
#define FIFO_W2dense3 17
#define ACTOR_READB_DENSE_dense3 19
#define FIFO_B2dense3 18
#define ACTOR_DENSE_dense3 20
#define FIFO_RELU2dense3 19
#define ACTOR_RELUDEN_dense3 21
#define FIFO_dense32RELU  20
#define ACTOR_READW_DENSE_dense4 22
#define FIFO_W2dense4 21
#define ACTOR_READB_DENSE_dense4 23
#define FIFO_B2dense4 22
#define ACTOR_DENSE_dense4 24
#define FIFO_RELU2dense4 23
#define FIFO_dense42SOFTMAX  24
#define ACTOR_SOFTMAX 25
#define FIFO_SM2WO 25
#define ACTOR_WRITEOUT 26
#define ACTOR_COUNT 27
#define FIFO_COUNT 26
struct _lide_c_cnnmulti_graph_context_struct;
typedef struct _lide_c_cnnmulti_graph_context_struct lide_c_cnnmulti_graph_context_type;
lide_c_cnnmulti_graph_context_type *lide_c_cnnmulti_graph_new(const char * InputsFile, 
    const char* Wconv1, const char* Bconv1,
    const char* Wconv2, const char* Bconv2,
    const char* Wdense1, const char* Bdense1,
    const char* Wdense2, const char* Bdense2,
    const char* Wdense3, const char* Bdense3,
    const char* Wdense4, const char* Bdense4,
    const char* OutFile );
void lide_c_cnnmulti_graph_terminate(lide_c_cnnmulti_graph_context_type *graph);
void lide_c_cnnmulti_graph_scheduler(lide_c_cnnmulti_graph_context_type *graph);
#endif
