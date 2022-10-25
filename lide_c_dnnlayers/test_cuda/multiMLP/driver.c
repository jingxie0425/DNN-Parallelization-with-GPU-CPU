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

#include "lide_c_multiMLP_graph.h"


int main(int argc, char **argv) {
    
    const char* InputsFile = "../../../extractParas_mod1/testInputX.csv";
    const char* WDense1 = "../../../extractParas_mod1/dense1Weight.csv";
    const char* BDense1 = "../../../extractParas_mod1/dense1Bias.csv";
    const char* WDense2 = "../../../extractParas_mod1/dense2Weight.csv";
    const char* BDense2 = "../../../extractParas_mod1/dense2Bias.csv";
    const char* WDense3 = "../../../extractParas_mod1/dense3Weight.csv";
    const char* BDense3 = "../../../extractParas_mod1/dense3Bias.csv";
    const char* WDense4 = "../../../extractParas_mod1/dense4Weight.csv";
    const char* BDense4 = "../../../extractParas_mod1/dense4Bias.csv";
            
    const char* OutFile = "../result.csv";

    lide_c_graph_context_type *graph = NULL;

    /* Create graph*/                               
    graph = (lide_c_graph_context_type*)lide_c_multiMLP_graph_new(InputsFile, WDense1, BDense1, WDense2, 
        BDense2, WDense3, BDense3, WDense4, BDense4, OutFile );  
    //execute graph
    graph->scheduler(graph);

    return 0;
}