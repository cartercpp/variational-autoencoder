//
// Created by cartercpp on 10/9/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <cstddef>
#include "matrix.hpp"
#include "math_vector.hpp"
#include "activation_functions.hpp"

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        const std::initializer_list<std::size_t> neuronsPerLayer,
        double learningRate,
        ACTIVATION_FUNCTION intermediateActivationFunction,
        ACTIVATION_FUNCTION outputActivationFunction,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(neuronsPerLayer.size() >= 3);

    // METHODS

    [[nodiscard]] std::pmr::vector<math_vector<double>> forward(const math_vector<double>& input) const
        pre(input.size() == m_neuronsPerLayer[0])
        post(activations : activations.size() == m_neuronsPerLayer.size());

    [[nodiscard]] math_vector<double> predict(const math_vector<double>& input) const
        pre(input.size() == m_neuronsPerLayer[0])
        post(prediction : prediction.size() == m_neuronsPerLayer.back());

    math_vector<double> backward(
        const std::pmr::vector<math_vector<double>>& activations,
        math_vector<double> gradient,
        bool gradientIsWrtPreActivation
    )
        pre(activations.size() == m_neuronsPerLayer.size())
        pre(gradient.size() == m_neuronsPerLayer.back())
        post(gradientWrtInput : gradientWrtInput.size() == m_neuronsPerLayer[0]);

    void update_weights_and_biases();
    void zero_gradients();

private:

    [[nodiscard]] static math_vector<double> Relu(math_vector<double>);
    [[nodiscard]] static math_vector<double> ReluDerivative(math_vector<double>);
    [[nodiscard]] static math_vector<double> Sigmoid(math_vector<double>);
    [[nodiscard]] static math_vector<double> SigmoidDerivative(math_vector<double>);

    std::pmr::vector<matrix<double>> m_weightMatrices,
                                     m_weightGradients;
    std::pmr::vector<math_vector<double>> m_biasVectors,
                                          m_biasGradients;
    std::pmr::vector<std::size_t> m_neuronsPerLayer;
    double m_learningRate;
    ACTIVATION_FUNCTION m_intermediateActivationFunction,
                        m_outputActivationFunction;
    mutable std::pmr::memory_resource* m_resource;
};