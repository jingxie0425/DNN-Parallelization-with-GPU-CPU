#ifndef _lide_c_graph_h
#define _lide_c_graph_h

/*******************************************************************************
@ddblock_begin copyright

Copyright (c) 1997-2017
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
#include "lide_c_basic.h"
#include "lide_c_actor.h"
//#include "../fifo/lide_c_fifo.h"
#include "lide_c_fifo_basic.h"
#include "lide_c_graph_def.h"

#define GRAPH_IN_CONN_DIRECTION  0   /* input port of an actor*/
#define GRAPH_OUT_CONN_DIRECTION 1   /* output port of an actor*/


/*****************************************************************************
A pointer to a "lide_c_graph_scheduler", which is a function that execute the 
dataflow graph
*****************************************************************************/
typedef void (*lide_c_graph_scheduler_ftype) 
        (struct lide_c_graph_context_struct *graph);  

/*****************************************************************************
Adds an actor in the graph: update the actor count and other parameters
*****************************************************************************/
void lide_c_graph_add_actor(struct lide_c_graph_context_struct *graph); 
        
/*****************************************************************************
Set connections between actors in the graph: set up the portrefs arrays, 
as well as data structures such as the source and sink tables.  
*****************************************************************************/
void lide_c_graph_add_connection(
        struct lide_c_graph_context_struct *graph, 
        struct lide_c_actor_context_struct *context, int port_index, 
        int direction);        

/*****************************************************************************
Update connections between actors in the graph: set up the portrefs arrays, 
as well as data structures such as the source and sink tables.  
*****************************************************************************/
void lide_c_graph_update_connection(
    struct lide_c_graph_context_struct *graph, 
        struct lide_c_actor_context_struct *context, int port_index,  
        int fifo_index, int direction);  

        
/*****************************************************************************
Set a actor in the graph.
*****************************************************************************/
void lide_c_graph_set_actor(struct lide_c_graph_context_struct *graph, 
        int index, lide_c_actor_context_type *actor_context);   
        
/*****************************************************************************
Get a actor context in the graph.
*****************************************************************************/
lide_c_actor_context_type *lide_c_graph_get_actor(
        struct lide_c_graph_context_struct *graph, int index);       
        
/*****************************************************************************
Set a fifo pointer in the graph.
*****************************************************************************/
void lide_c_graph_set_fifo(struct lide_c_graph_context_struct *graph, 
        int index, lide_c_fifo_pointer fifo);         
        
/*****************************************************************************
Get a fifo pointer in the graph.
*****************************************************************************/
lide_c_fifo_pointer lide_c_graph_get_fifo(
        struct lide_c_graph_context_struct *graph, int index);  
        
/*****************************************************************************
Set actor count in the graph.
*****************************************************************************/
void lide_c_graph_set_actor_count(
        struct lide_c_graph_context_struct *graph, int count);           

/*****************************************************************************
Get actor count in the graph.
*****************************************************************************/
int lide_c_graph_get_actor_count(struct lide_c_graph_context_struct *graph);  

/*****************************************************************************
Set fifo count in the graph.
*****************************************************************************/
void lide_c_graph_set_fifo_count(
        struct lide_c_graph_context_struct *graph, int count);   

/*****************************************************************************
Get fifo count in the graph.
*****************************************************************************/
int lide_c_graph_get_fifo_count(struct lide_c_graph_context_struct *graph); 

 /*****************************************************************************
Set scheduler function pointer in the graph.
*****************************************************************************/
void lide_c_graph_set_scheduler(struct lide_c_graph_context_struct *graph, 
        lide_c_graph_scheduler_ftype scheduler_ftype);   

/*****************************************************************************
Get scheduler function pointer in the graph.
*****************************************************************************/
lide_c_graph_scheduler_ftype lide_c_graph_get_scheduler(
        struct lide_c_graph_context_struct *graph); 

/*****************************************************************************
Get sink array pointer in the graph.
*****************************************************************************/
lide_c_actor_context_type **lide_c_graph_get_sink_array(
        struct lide_c_graph_context_struct *graph); 

/*****************************************************************************
Get sink array pointer in the graph.
*****************************************************************************/
lide_c_actor_context_type **lide_c_graph_get_source_array(
        struct lide_c_graph_context_struct *graph); 
        
struct lide_c_graph_context_struct {
#include "lide_c_graph_context_type_common.h"
};


#endif
