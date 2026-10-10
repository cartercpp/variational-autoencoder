//
// Created by cartercpp on 10/9/26.
//

#include "neural_network.hpp"
#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <random>
#include <utility>
#include <cstddef>
#include <cmath>
#include "activation_functions.hpp"
#include "matrix.hpp"
#include "math_vector.hpp"

neural_network::neural_network(
    const std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate,
    ACTIVATION_FUNCTION intermediateActivationFunction,
    ACTIVATION_FUNCTION outputActivationFunction,
    std::pmr::memory_resource* resource
)
    : m_weightMatrices{resource},
    m_weightGradients{resource},
    m_biasVectors{resource},
    m_biasGradients{resource},
    m_neuronsPerLayer{resource},
    m_learningRate{learningRate},
    m_intermediateActivationFunction{intermediateActivationFunction},
    m_outputActivationFunction{outputActivationFunction},
    m_resource{resource}
{
    m_neuronsPerLayer.reserve(neuronsPerLayer.size());
    for (const std::size_t neurons : neuronsPerLayer)
        m_neuronsPerLayer.push_back(neurons);

    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(-1, 1);

    auto getRandomWeight = [&](std::size_t layer) {
        if (layer + 1 < m_neuronsPerLayer.size())
            switch (m_intermediateActivationFunction)
            {
                case ACTIVATION_FUNCTION::RELU:
                    return std::normal_distribution(0.0, std::sqrt(2.0 / m_neuronsPerLayer[layer - 1]))(rng);
                default:
                    return dist(rng);
            }
        else
            switch (m_outputActivationFunction)
            {
                case ACTIVATION_FUNCTION::RELU:
                    return std::normal_distribution(0.0, std::sqrt(2.0 / m_neuronsPerLayer[layer - 1]))(rng);
                default:
                    return std::normal_distribution(
                        0.0,
                        std::sqrt(2.0 / (m_neuronsPerLayer[layer - 1] + m_neuronsPerLayer[layer]))
                        )(rng);
            }
    };

    m_weightMatrices.reserve(m_neuronsPerLayer.size() - 1);
    m_weightGradients.reserve(m_neuronsPerLayer.size() - 1);

    for (std::size_t layer = 1; layer < m_neuronsPerLayer.size(); ++layer)
    {
        const std::size_t layerSize = m_neuronsPerLayer[layer],
                          prevLayerSize = m_neuronsPerLayer[layer - 1];

        matrix<double> weights(layerSize, prevLayerSize, 0, m_resource);
        for (std::size_t i = 0; i < layerSize; ++i)
            for (std::size_t i2 = 0; i2 < prevLayerSize; ++i2)
                weights[i][i2] = getRandomWeight(layer);

        m_weightMatrices.emplace_back(std::move(weights));
        m_weightGradients.emplace_back(layerSize, prevLayerSize, 0, m_resource);
        m_biasVectors.emplace_back(layerSize, 0, m_resource);
        m_biasGradients.emplace_back(layerSize, 0, m_resource);
    }
}

math_vector<double> neural_network::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) * vec[i];

    return vec;
}

math_vector<double> neural_network::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

math_vector<double> neural_network::Sigmoid(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = 1 / (1 + std::exp(-vec[i]));

    return vec;
}

math_vector<double> neural_network::SigmoidDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] * (1 - vec[i]);

    return vec;
}

std::pmr::vector<math_vector<double>> neural_network::forward(const math_vector<double>& input) const
{
    std::pmr::vector<math_vector<double>> activations{m_resource};
    activations.reserve(m_neuronsPerLayer.size());
    activations.push_back(input);

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        math_vector<double> preactivations{m_weightMatrices[i] * activations[i] + m_biasVectors[i]};

        if (i + 1 < m_weightMatrices.size())
            switch (m_intermediateActivationFunction)
            {
                case ACTIVATION_FUNCTION::RELU:
                    activations.emplace_back(Relu(std::move(preactivations)));
                    break;
                case ACTIVATION_FUNCTION::SIGMOID:
                    activations.emplace_back(Sigmoid(std::move(preactivations)));
                    break;
                case ACTIVATION_FUNCTION::LINEAR:
                    activations.emplace_back(std::move(preactivations));
                    break;
            }
        else
            switch (m_outputActivationFunction)
            {
                case ACTIVATION_FUNCTION::RELU:
                    activations.emplace_back(Relu(std::move(preactivations)));
                    break;
                case ACTIVATION_FUNCTION::SIGMOID:
                    activations.emplace_back(Sigmoid(std::move(preactivations)));
                    break;
                case ACTIVATION_FUNCTION::LINEAR:
                    activations.emplace_back(std::move(preactivations));
                    break;
            }
    }

    return activations;
}

math_vector<double> neural_network::predict(const math_vector<double>& input) const
{
    return forward(input).back();
}

math_vector<double> neural_network::backward(
    const std::pmr::vector<math_vector<double>>& activations,
    math_vector<double> gradient,
    bool gradientIsWrtPreActivation
)
{
    for (std::size_t layer = 0; layer < activations.size(); ++layer)
        if (activations[layer].size() != m_neuronsPerLayer[layer])
            throw std::invalid_argument{"`activations` doesn't fit NN configuration"};

    for (std::size_t i = m_weightMatrices.size(); i-- > 0;)
    {
        if (i == m_weightMatrices.size() - 1)
        {
            if (!gradientIsWrtPreActivation)
                switch (m_outputActivationFunction)
                {
                    case ACTIVATION_FUNCTION::RELU:
                        gradient = gradient.multiply(ReluDerivative(activations[i + 1]));
                        break;
                    case ACTIVATION_FUNCTION::SIGMOID:
                        gradient = gradient.multiply(SigmoidDerivative(activations[i + 1]));
                        break;
                }
        }
        else
            switch (m_intermediateActivationFunction)
            {
                case ACTIVATION_FUNCTION::RELU:
                    gradient = gradient.multiply(ReluDerivative(activations[i + 1]));
                    break;
                case ACTIVATION_FUNCTION::SIGMOID:
                    gradient = gradient.multiply(SigmoidDerivative(activations[i + 1]));
                    break;
            }

        m_weightGradients[i] = outer_product(gradient, activations[i]);
        m_biasGradients[i] = gradient;
        gradient = gradient * m_weightMatrices[i];
    }

    return gradient;
}

void neural_network::update_weights_and_biases()
{
    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        m_weightMatrices[i] -= m_learningRate * m_weightGradients[i];
        m_biasVectors[i] -= m_learningRate * m_biasGradients[i];
    }

    zero_gradients();
}

void neural_network::zero_gradients()
{
    for (matrix<double>& weightGradientRef : m_weightGradients)
        for (std::size_t i = 0; i < weightGradientRef.rows(); ++i)
            for (std::size_t i2 = 0; i2 < weightGradientRef.columns(); ++i2)
                weightGradientRef[i][i2] = 0;

    for (math_vector<double>& biasGradientRef : m_biasGradients)
        for (std::size_t i = 0; i < biasGradientRef.size(); ++i)
            biasGradientRef[i] = 0;
}