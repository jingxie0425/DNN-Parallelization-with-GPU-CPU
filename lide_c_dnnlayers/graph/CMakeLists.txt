cmake_minimum_required(VERSION 3.14)

project(CNN C)

add_library(graph libgraph.a)
target_include_directories(graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../actor)
