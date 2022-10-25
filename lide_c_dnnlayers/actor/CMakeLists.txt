cmake_minimum_required(VERSION 3.14)

project(CNN C)

add_library(actor libactor.a)
target_include_directories(actor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
