cmake_minimum_required(VERSION 4.3)
project(VariationalAutoencoder)

set(CMAKE_CXX_STANDARD 26)

add_executable(VariationalAutoencoder main.cpp
        math_vector.hpp
        matrix.hpp
        neural_network.hpp
        neural_network.cpp
        variational_autoencoder.hpp
        variational_autoencoder.cpp
        activation_functions.hpp)
