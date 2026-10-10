//
// Created by cartercpp on 10/9/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <span>
#include <memory_resource>
#include <simd>
#include <concepts>
#include <cstddef>

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class math_vector
{
public:

    // CONSTRUCTORS

    explicit math_vector(
        std::size_t size,
        ValueType value = 0,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource}, m_resource{resource}
    {
        m_data.resize(size, value);
    }

    explicit math_vector(
        std::initializer_list<ValueType> data,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource}, m_resource{resource}
    {
        m_data.reserve(data.size());
        for (const ValueType value : data)
            m_data.push_back(value);
    }

    // METHODS

    [[nodiscard]] std::size_t size() const noexcept
    {
        return m_data.size();
    }

    [[nodiscard]] ValueType& operator[](const std::size_t index)
        pre(index < m_data.size())
    {
        return m_data[index];
    }

    [[nodiscard]] const ValueType& operator[](const std::size_t index) const
        pre(index < m_data.size())
    {
        return m_data[index];
    }

    math_vector subvector(const std::size_t start, const std::size_t end) const
        pre(start < end)
        pre(end <= m_data.size())
    {
        math_vector output(end - start, 0);
        for (std::size_t i = 0; i < end - start; ++i)
            output[i] = m_data[start + i];

        return output;
    }

    math_vector concatenate(const math_vector& other) const
    {
        math_vector output(m_data.size() + other.size(), 0, m_resource);

        for (std::size_t i = 0; i < m_data.size(); ++i)
            output[i] = m_data[i];

        for (std::size_t i = 0; i < other.size(); ++i)
            output[m_data.size() + i] = other[i];

        return output;
    }

    math_vector multiply(const math_vector& other) const
        pre(m_data.size() == other.size())
    {
        constexpr std::size_t intervalSize = 5;
        using V = std::simd::vec<ValueType, intervalSize>;

        math_vector output(m_data.size(), 0, m_resource);
        for (std::size_t start = 0; start < m_data.size(); start += intervalSize)
        {
            const auto a = std::span{m_data}.subspan(start);
            const auto b = std::span{other.m_data}.subspan(start);

            const V lArg{std::simd::partial_load<V>(a)},
                    rArg{std::simd::partial_load<V>(b)};

            const V result{lArg * rArg};
            for (std::size_t i = 0; (i < intervalSize) && (start + i < m_data.size()); ++i)
                output[start + i] = result[i];
        }

        return output;
    }

    math_vector& operator*=(ValueType scalar)
    {
        constexpr std::size_t intervalSize = 5;
        using V = std::simd::vec<ValueType, intervalSize>;

        for (std::size_t start = 0; start < m_data.size(); start += intervalSize)
        {
            const auto sp = std::span{m_data}.subspan(start);
            const V vec{std::simd::partial_load<V>(sp)};

            const V result{vec * scalar};
            for (std::size_t i = 0; (i < intervalSize) && (start + i < m_data.size()); ++i)
                m_data[start + i] = result[i];
        }

        return *this;
    }

    math_vector& operator/=(const ValueType scalar)
        pre(scalar != 0)
    {
        constexpr std::size_t intervalSize = 5;
        using V = std::simd::vec<ValueType, intervalSize>;

        for (std::size_t start = 0; start < m_data.size(); start += intervalSize)
        {
            const auto sp = std::span{m_data}.subspan(start);
            const V vec{std::simd::partial_load<V>(sp)};

            const V result{vec / scalar};
            for (std::size_t i = 0; (i < intervalSize) && (start + i < m_data.size()); ++i)
                m_data[start + i] = result[i];
        }

        return *this;
    }

    math_vector& operator+=(const math_vector& other)
        pre(m_data.size() == other.size())
    {
        constexpr std::size_t intervalSize = 5;
        using V = std::simd::vec<ValueType, intervalSize>;

        for (std::size_t start = 0; start < m_data.size(); start += intervalSize)
        {
            const auto a = std::span{m_data}.subspan(start);
            const auto b = std::span{other.m_data}.subspan(start);

            const V lArg{std::simd::partial_load<V>(a)},
                    rArg{std::simd::partial_load<V>(b)};

            const V result{lArg + rArg};
            for (std::size_t i = 0; (i < intervalSize) && (start + i < m_data.size()); ++i)
                m_data[start + i] = result[i];
        }

        return *this;
    }

    math_vector& operator-=(const math_vector& other)
        pre(m_data.size() == other.size())
    {
        constexpr std::size_t intervalSize = 5;
        using V = std::simd::vec<ValueType, intervalSize>;

        for (std::size_t start = 0; start < m_data.size(); start += intervalSize)
        {
            const auto a = std::span{m_data}.subspan(start);
            const auto b = std::span{other.m_data}.subspan(start);

            const V lArg{std::simd::partial_load<V>(a)},
                    rArg{std::simd::partial_load<V>(b)};

            const V result{lArg - rArg};
            for (std::size_t i = 0; (i < intervalSize) && (start + i < m_data.size()); ++i)
                m_data[start + i] = result[i];
        }

        return *this;
    }

    ValueType operator*(const math_vector& other) const
        pre(m_data.size() == other.size())
    {
        constexpr std::size_t intervalSize = 5;
        using V = std::simd::vec<ValueType, intervalSize>;

        ValueType output = 0;
        for (std::size_t start = 0; start < m_data.size(); start += intervalSize)
        {
            const auto a = std::span{m_data}.subspan(start);
            const auto b = std::span{other.m_data}.subspan(start);

            const V lArg{std::simd::partial_load<V>(a)},
                    rArg{std::simd::partial_load<V>(b)};

            output += std::simd::reduce(lArg * rArg);
        }

        return output;
    }

    [[nodiscard]] bool operator==(const math_vector& other) const
    {
        if (m_data.size() != other.size())
            return false;

        constexpr std::size_t intervalSize = 5;
        using V = std::simd::vec<ValueType, intervalSize>;

        for (std::size_t start = 0; start < m_data.size(); start += intervalSize)
        {
            const auto a = std::span{m_data}.subspan(start);
            const auto b = std::span{other.m_data}.subspan(start);

            const V lArg{std::simd::partial_load<V>(a)},
                    rArg{std::simd::partial_load<V>(b)};

            const auto cmp{lArg == rArg};
            if (!std::simd::all_of(cmp))
                return false;
        }

        return true;
    }

    [[nodiscard]] bool operator!=(const math_vector& other) const
    {
        return !(*this == other);
    }

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept
    {
        return m_resource;
    }

private:

    std::pmr::vector<ValueType> m_data;
    mutable std::pmr::memory_resource* m_resource;
};

template <typename ValueType>
[[nodiscard]] auto operator*(math_vector<ValueType> vec, ValueType scalar)
{
    return vec *= scalar;
}

template <typename ValueType>
[[nodiscard]] auto operator*(ValueType scalar, math_vector<ValueType> vec)
{
    return vec *= scalar;
}

template <typename ValueType>
[[nodiscard]] auto operator/(math_vector<ValueType> vec, ValueType scalar)
{
    return vec /= scalar;
}

template <typename ValueType>
[[nodiscard]] auto operator+(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
[[nodiscard]] auto operator-(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg -= rArg;
}