//
// Created by cartercpp on 10/9/26.
//

#pragma once

#include <initializer_list>
#include <memory_resource>
#include <cstddef>
#include "math_vector.hpp"
#include "neural_network.hpp"

class variational_autoencoder
{
public:

    // CONSTRUCTORS

    explicit variational_autoencoder(
        const std::initializer_list<std::size_t> encoderNeuronsPerLayer,
        double encoderLearningRate,
        const std::initializer_list<std::size_t> decoderNeuronsPerLayer,
        double decoderLearningRate,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(*encoderNeuronsPerLayer.begin() == *(decoderNeuronsPerLayer.begin() + decoderNeuronsPerLayer.size() - 1))
        pre(*(encoderNeuronsPerLayer.begin() + encoderNeuronsPerLayer.size() - 1) / 2 == *decoderNeuronsPerLayer.begin())
        pre(*(encoderNeuronsPerLayer.begin() + encoderNeuronsPerLayer.size() - 1) % 2 == 0);

    // METHODS

    math_vector<double> encode(const math_vector<double>& input) const;
    math_vector<double> decode(const math_vector<double>& latentVector) const;
    math_vector<double> regenerate(const math_vector<double>& input) const;
    void train(const math_vector<double>& input);

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept;

private:

    neural_network m_encoder,
                   m_decoder;
    mutable std::pmr::memory_resource* m_resource;
};