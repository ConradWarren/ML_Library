cmake_minimum_required(VERSION 3.26)
project(ML_Library)

set(CMAKE_CXX_STANDARD 17)

add_library(ML_Library layer.h
        dense_layer.cpp
        convolutional_layer.cpp
        max_pooling_layer.cpp
        model.cpp
        model.h
)
