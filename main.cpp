

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "variational_autoencoder.hpp"

namespace {

constexpr std::size_t side = 28;
constexpr std::size_t pixelsPerImage = side * side;
constexpr std::size_t latentSize = 16;
constexpr std::size_t displayInterval = 500;

struct mnist_images {
    std::size_t count;
    std::vector<std::uint8_t> pixels;
};

std::uint32_t read_u32(std::ifstream& stream)
{
    std::array<unsigned char, 4> bytes{};
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), 4))
        throw std::runtime_error("Invalid MNIST header");

    return (std::uint32_t(bytes[0]) << 24) |
           (std::uint32_t(bytes[1]) << 16) |
           (std::uint32_t(bytes[2]) << 8) |
            std::uint32_t(bytes[3]);
}

mnist_images load_mnist(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Could not open " + path.string());

    const auto magic = read_u32(stream);
    const auto count = read_u32(stream);
    const auto rows = read_u32(stream);
    const auto columns = read_u32(stream);

    if (magic != 2051 || rows != side || columns != side || count == 0)
        throw std::runtime_error("Expected uncompressed MNIST 28x28 images");

    mnist_images result{
        count, std::vector<std::uint8_t>(std::size_t(count) * pixelsPerImage)
    };
    if (!stream.read(reinterpret_cast<char*>(result.pixels.data()),
                     static_cast<std::streamsize>(result.pixels.size())))
        throw std::runtime_error("MNIST image file is truncated");

    return result;
}

math_vector<double> image_at(const mnist_images& images, std::size_t index)
{
    math_vector<double> image(pixelsPerImage);
    const std::size_t offset = index * pixelsPerImage;

    for (std::size_t i = 0; i < pixelsPerImage; ++i)
        image[i] = images.pixels[offset + i] / 255.0;

    return image;
}

int brightness(double value)
{
    if (!std::isfinite(value))
        throw std::runtime_error("Network produced NaN or infinity");
    return static_cast<int>(std::lround(255.0 * std::clamp(value, 0.0, 1.0)));
}

// Two square, true-color panels side by side, fitting in 80 terminal columns.
// Each Unicode upper-half block displays TWO vertically adjacent pixels,
// so a 38x38 image occupies just 38 columns and 19 terminal rows.
constexpr std::size_t panelWidth = 38;
constexpr std::size_t panelRows = panelWidth / 2;
constexpr std::size_t panelGap = 3;
static_assert(2 * panelWidth + panelGap <= 80);

// Bilinear interpolation when resizing the 28x28 MNIST image for display.
double sample_image(const math_vector<double>& image, std::size_t x, std::size_t y)
{
    const auto position = [](std::size_t pixel) {
        return std::clamp((static_cast<double>(pixel) + 0.5) * side / panelWidth - 0.5,
                          0.0, static_cast<double>(side - 1));
    };

    const double sx = position(x), sy = position(y);
    const auto x0 = static_cast<std::size_t>(sx);
    const auto y0 = static_cast<std::size_t>(sy);
    const auto x1 = std::min(x0 + 1, side - 1);
    const auto y1 = std::min(y0 + 1, side - 1);
    const double dx = sx - x0, dy = sy - y0;

    return (1 - dy) * ((1 - dx) * image[y0 * side + x0] +
                      dx * image[y0 * side + x1]) +
           dy * ((1 - dx) * image[y1 * side + x0] +
                 dx * image[y1 * side + x1]);
}

void draw_panel_row(const math_vector<double>& image, std::size_t row)
{
    for (std::size_t x = 0; x < panelWidth; ++x) {
        const int top = brightness(sample_image(image, x, 2 * row));
        const int bottom = brightness(sample_image(image, x, 2 * row + 1));

        // Foreground colors the upper half, background colors the lower half.
        std::cout << "\x1b[38;2;" << top << ';' << top << ';' << top
                  << "m\x1b[48;2;" << bottom << ';' << bottom << ';' << bottom
                  << "m▀";
    }
    std::cout << "\x1b[0m";
}

void clear_screen()
{
    std::cout << "\x1b[2J\x1b[H";
}

double reconstruction_bce(const math_vector<double>& input,
                          const math_vector<double>& reconstruction)
{
    constexpr double small = 1e-7;
    double sum = 0.0;
    for (std::size_t i = 0; i < pixelsPerImage; ++i) {
        const double x = input[i];
        const double p = std::clamp(reconstruction[i], small, 1.0 - small);
        sum -= x * std::log(p) + (1.0 - x) * std::log(1.0 - p);
    }
    return sum / pixelsPerImage;
}

void show_comparison(const variational_autoencoder& vae,
                     const math_vector<double>& original,
                     const std::string& status)
{
    // The displayed original is exactly the input to regenerate().
    const auto reconstruction = vae.regenerate(original);

    clear_screen();
    std::cout << status << "  |  BCE/pixel: "
              << reconstruction_bce(original, reconstruction) << "\n\n";
    std::cout << "Original" << std::string(panelWidth + panelGap - 8, ' ')
              << "Reconstruction\n\n";

    for (std::size_t row = 0; row < panelRows; ++row) {
        draw_panel_row(original, row);
        std::cout << std::string(panelGap, ' ');
        draw_panel_row(reconstruction, row);
        std::cout << '\n';
    }
    std::cout << "\nCtrl+C to stop\n" << std::flush;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::filesystem::path path =
            argc > 1 ? argv[1] : "train-images-idx3-ubyte";
        const std::size_t trainingSteps =
            argc > 2 ? std::stoull(argv[2]) : 180000; // ~3 MNIST epochs

        std::cout << "Loading MNIST...\n";
        const mnist_images images = load_mnist(path);

        variational_autoencoder vae(
            {pixelsPerImage, 128, 64, 2 * latentSize}, 0.001,
            {latentSize, 64, 128, pixelsPerImage}, 0.001
        );

        std::mt19937 rng{std::random_device{}()};

        // Shuffle every epoch instead of sampling training images with replacement.
        std::vector<std::size_t> order(images.count);
        std::iota(order.begin(), order.end(), 0);

        for (std::size_t step = 0; step < trainingSteps; ++step) {
            if (step % images.count == 0)
                std::shuffle(order.begin(), order.end(), rng);

            // Use the exact same MNIST image for training and the display.
            const auto original = image_at(images, order[step % images.count]);
            vae.train(original);

            const std::size_t done = step + 1;
            if (done == 1 || done % displayInterval == 0 || done == trainingSteps)
                show_comparison(vae, original, "Train " + std::to_string(done)
                                                    + '/' + std::to_string(trainingSteps));
        }

        // After training, cycle through real MNIST images and reconstruct each one.
        for (std::size_t frame = 0;; ++frame) {
            const std::size_t index = frame % images.count;
            show_comparison(vae, image_at(images, index),
                            "MNIST #" + std::to_string(index));
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        }
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'
                  << "Usage: ./vae [train-images-idx3-ubyte] [training_steps]\n";
        return 1;
    }
}

