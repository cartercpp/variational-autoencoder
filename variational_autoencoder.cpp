//
// Created by cartercpp on 10/9/26.
//

#include "variational_autoencoder.hpp"
#include <initializer_list>
#include <memory_resource>
#include <random>
#include <cmath>
#include <cstddef>
#include "math_vector.hpp"
#include "neural_network.hpp"
#include "activation_functions.hpp"

variational_autoencoder::variational_autoencoder(
    const std::initializer_list<std::size_t> encoderNeuronsPerLayer,
    double encoderLearningRate,
    const std::initializer_list<std::size_t> decoderNeuronsPerLayer,
    double decoderLearningRate,
    std::pmr::memory_resource* resource
)
    : m_encoder{encoderNeuronsPerLayer, encoderLearningRate, ACTIVATION_FUNCTION::RELU, ACTIVATION_FUNCTION::LINEAR, resource},
    m_decoder{decoderNeuronsPerLayer, decoderLearningRate, ACTIVATION_FUNCTION::RELU, ACTIVATION_FUNCTION::SIGMOID, resource},
    m_resource{resource}
{}

void variational_autoencoder::train(const math_vector<double>& input)
{
    // compute latent vector:
    const std::pmr::vector<math_vector<double>> encoderActivations{m_encoder.forward(input)};
    const math_vector<double> temp{encoderActivations.back()},
                              mean{temp.subvector(0, temp.size() / 2)},
                              logVariance{temp.subvector(temp.size() / 2, temp.size())};

    std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<double> dist(0, 1);

    math_vector<double> epsilon(mean.size(), 0, m_resource),
                        latentVector(mean.size(), 0, m_resource);
    for (std::size_t i = 0; i < mean.size(); ++i)
    {
        epsilon[i] = dist(rng);
        latentVector[i] = mean[i] + std::exp(0.5 * logVariance[i]) * epsilon[i];
    }

    // compute reconstruction and error:
    const std::pmr::vector<math_vector<double>> decoderActivations{m_decoder.forward(latentVector)};
    const math_vector<double>& reconstruction{decoderActivations.back()};
    const math_vector<double> reconstructionGradient{reconstruction - input};

    // backpropagate into decoder:
    const math_vector<double> decoderGradient{m_decoder.backward(decoderActivations, reconstructionGradient, true)};

    // backpropagate into encoder:
    const math_vector<double> meanGradient{decoderGradient + mean};

    math_vector<double> logVarianceGradient(logVariance.size(), 0, m_resource);
    for (std::size_t i = 0; i < logVariance.size(); ++i)
        logVarianceGradient[i] = 0.5 * decoderGradient[i] * std::exp(0.5 * logVariance[i]) * epsilon[i]
                                 + 0.5 * (std::exp(logVariance[i]) - 1);

    const math_vector<double> encoderGradient{meanGradient.concatenate(logVarianceGradient)};
    m_encoder.backward(encoderActivations, encoderGradient, false);

    // update weights:
    m_encoder.update_weights_and_biases();
    m_decoder.update_weights_and_biases();
}

math_vector<double> variational_autoencoder::encode(const math_vector<double>& input) const
{
    const math_vector<double> temp{m_encoder.predict(input)},
                              mean{temp.subvector(0, temp.size() / 2)},
                              logVariance{temp.subvector(temp.size() / 2, temp.size())};

    std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<double> dist(0, 1);

    math_vector<double> latentVector(mean.size(), 0, m_resource);
    for (std::size_t i = 0; i < mean.size(); ++i)
        latentVector[i] = mean[i] + std::exp(0.5 * logVariance[i]) * dist(rng);

    return latentVector;
}

math_vector<double> variational_autoencoder::decode(const math_vector<double>& latentVector) const
{
    return m_decoder.predict(latentVector);
}

math_vector<double> variational_autoencoder::regenerate(const math_vector<double>& input) const
{
    return decode(encode(input));
}

std::pmr::memory_resource* variational_autoencoder::resource() const noexcept
{
    return m_resource;
}