/*******************************************************************************
@ddblock_begin copyright

Copyright (c) 1997-2019
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
#include <unistd.h>
#include "lide_c_util.h"

#include "lide_c_multiCNN_graph.h"


int main(int argc, char **argv) {
    const char* WC1 = "../../../extractParas_mod2_pruned/cnn1Weight.csv";
    const char* BC1 = "../../../extractParas_mod2_pruned/cnn1Bias.csv";
    const char* WC2 = "../../../extractParas_mod2_pruned/cnn2Weight.csv";
    const char* BC2 = "../../../extractParas_mod2_pruned/cnn2Bias.csv";
    const char* InputsFile = "../../../extractParas_mod2_pruned/testInputX.csv";

    const char* WFD = "../../../extractParas_mod2_pruned/dense1Weight.csv";
    const char* BFD = "../../../extractParas_mod2_pruned/dense1Bias.csv";
    const char* WD2 = "../../../extractParas_mod2_pruned/dense2Weight.csv";
    const char* BD2 = "../../../extractParas_mod2_pruned/dense2Bias.csv";
    const char* WD3 = "../../../extractParas_mod2_pruned/dense3Weight.csv";
    const char* BD3 = "../../../extractParas_mod2_pruned/dense3Bias.csv";
    const char* WD4 = "../../../extractParas_mod2_pruned/dense4Weight.csv";
    const char* BD4 = "../../../extractParas_mod2_pruned/dense4Bias.csv";
    

    const char* OutFile = "../result.csv";

    lide_c_graph_context_type *graph = NULL;

    /* Create graph*/                               
    graph = (lide_c_graph_context_type*)lide_c_multiCNN_graph_new(WC1,BC1, 
    InputsFile,WC2,BC2,WFD,BFD,WD2,BD2,WD3,BD3,WD4,BD4,OutFile);
    //execute graph
    graph->scheduler(graph);

    return 0;
}