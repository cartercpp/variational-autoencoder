//
// Created by cartercpp on 10/9/26.
//

#pragma once

#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <concepts>
#include <cstddef>
#include "math_vector.hpp"

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class matrix
{
public:

    // CONSTRUCTORS

    explicit matrix(
        const std::size_t rows,
        const std::size_t columns,
        ValueType value = 0,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(rows > 0)
        pre(columns > 0)
        : m_data{resource}, m_rows{rows}, m_columns{columns}, m_resource{resource}
    {
        m_data.resize(rows, math_vector<ValueType>(columns, value, resource));
    }

    explicit matrix(
        const std::initializer_list<math_vector<ValueType>> data,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(data.size() > 0)
        pre(data.begin()->size() > 0)
        : m_data{resource}, m_rows{data.size()}, m_columns{data.begin()->size()}, m_resource{resource}
    {
        m_data.reserve(m_rows);
        for (const math_vector<ValueType>& rowVec : data)
        {
            if (rowVec.size() != m_columns)
                throw std::invalid_argument{"Each row must have the same # of columns"};

            m_data.push_back(rowVec);
        }
    }

    // METHODS

    [[nodiscard]] std::size_t rows() const noexcept
    {
        return m_rows;
    }

    [[nodiscard]] std::size_t columns() const noexcept
    {
        return m_columns;
    }

    [[nodiscard]] auto& operator[](const std::size_t row)
        pre(row < m_rows)
    {
        return m_data[row];
    }

    [[nodiscard]] const auto& operator[](const std::size_t row) const
        pre(row < m_rows)
    {
        return m_data[row];
    }

    math_vector<ValueType> get_column(const std::size_t column) const
        pre(column < m_columns)
    {
        math_vector<ValueType> output(m_rows, 0);
        for (std::size_t row = 0; row < m_rows; ++row)
            output[row] = m_data[row][column];

        return output;
    }

    matrix& operator+=(const matrix& other)
        pre(m_rows == other.rows())
        pre(m_columns == other.columns())
    {
        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] += other[row];

        return *this;
    }

    matrix& operator-=(const matrix& other)
        pre(m_rows == other.rows())
        pre(m_columns == other.columns())
    {
        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] -= other[row];

        return *this;
    }

    matrix& operator*=(ValueType scalar)
    {
        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] *= scalar;

        return *this;
    }

    matrix& operator/=(const ValueType scalar)
        pre(scalar != 0)
    {
        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] /= scalar;

        return *this;
    }

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept
    {
        return m_resource;
    }

private:

    std::pmr::vector<math_vector<ValueType>> m_data;
    std::size_t m_rows,
                m_columns;
    mutable std::pmr::memory_resource* m_resource;
};

template <typename ValueType>
bool operator==(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    if ((lArg.rows() != rArg.rows()) || (lArg.columns() != rArg.columns()))
        return false;

    for (std::size_t row = 0; row < lArg.rows(); ++row)
        if (lArg[row] != rArg[row])
            return false;

    return true;
}

template <typename ValueType>
bool operator!=(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    return !(lArg == rArg);
}

template <typename ValueType>
auto outer_product(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    matrix<ValueType> output(lArg.size(), rArg.size(), 0, lArg.resource());

    for (std::size_t i = 0; i < lArg.size(); ++i)
        for (std::size_t i2 = 0; i2 < rArg.size(); ++i2)
            output[i][i2] = lArg[i] * rArg[i2];

    return output;
}

template <typename ValueType>
auto operator*(matrix<ValueType> mat, ValueType scalar)
{
    return mat *= scalar;
}

template <typename ValueType>
auto operator*(ValueType scalar, matrix<ValueType> mat)
{
    return mat *= scalar;
}

template <typename ValueType>
auto operator/(matrix<ValueType> mat, ValueType scalar)
{
    return mat /= scalar;
}

template <typename ValueType>
auto operator+(matrix<ValueType> lArg, const matrix<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
auto operator-(matrix<ValueType> lArg, const matrix<ValueType>& rArg)
{
    return lArg -= rArg;
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
    pre(lArg.columns() == rArg.rows())
{
    matrix<ValueType> output(lArg.rows(), rArg.columns(), lArg.resource());

    for (std::size_t row = 0; row < lArg.rows(); ++row)
        for (std::size_t column = 0; column < rArg.columns(); ++column)
            output[row][column] = lArg[row] * rArg.get_column(column);

    return output;
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& mat, const math_vector<ValueType>& vec)
    pre(mat.columns() == vec.size())
{
    math_vector<ValueType> output(mat.rows(), 0, mat.resource());
    for (std::size_t i = 0; i < mat.rows(); ++i)
        output[i] = mat[i] * vec;

    return output;
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& vec, const matrix<ValueType>& mat)
    pre(vec.size() == mat.rows())
{
    math_vector<ValueType> output(mat.columns(), 0, vec.resource());
    for (std::size_t i = 0; i < mat.columns(); ++i)
        output[i] = vec * mat.get_column(i);

    return output;
}