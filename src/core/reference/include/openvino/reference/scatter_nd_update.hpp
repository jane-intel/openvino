// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <numeric>

#include "openvino/core/shape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/reference/maximum.hpp"
#include "openvino/reference/minimum.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/xor.hpp"
#include "utils/span.hpp"

namespace ov {
namespace reference {
using Reduction = ov::op::v15::ScatterNDUpdate::Reduction;
template <typename T>
using reduction_function = T (*)(const T, const T);

namespace func {
// TODO move this functions to other reference implementations to reduce binary size. Binary for
// ScatterElementsUpdate-12 can also be updated. Ticket: CVS-138266
template <class T>
constexpr T add(const T a, const T b) {
    return a + b;
}
template <class T>
constexpr T subtract(const T a, const T b) {
    return a - b;
}

template <class T>
constexpr T logical_and(const T a, const T b) {
    return static_cast<bool>(a) && static_cast<bool>(b);
}

template <class T>
constexpr T logical_or(const T a, const T b) {
    return static_cast<bool>(a) || static_cast<bool>(b);
}

}  // namespace func

template <typename T,
          typename std::enable_if<!std::is_same<typename std::decay<T>::type, char>::value>::type* = nullptr>
reduction_function<T> reduction_functor_for(const Reduction reduction_type) {
    using U = typename std::decay<T>::type;
    switch (reduction_type) {
    case Reduction::MAX:
        return func::max<U>;
    case Reduction::MIN:
        return func::min<U>;
    case Reduction::PROD:
        return func::multiply<U>;
    case Reduction::SUM:
        return func::add<U>;
    case Reduction::SUB:
        return func::subtract<U>;
    case Reduction::NONE:
    default:
        return nullptr;
    }
}

template <typename T, typename std::enable_if<std::is_same<typename std::decay<T>::type, char>::value>::type* = nullptr>
reduction_function<T> reduction_functor_for(const Reduction reduction_type) {
    using U = typename std::decay<T>::type;
    switch (reduction_type) {
    case Reduction::MIN:
    case Reduction::PROD:
        return func::logical_and<U>;
    case Reduction::SUM:
    case Reduction::MAX:
        return func::logical_or<U>;
    case Reduction::SUB:
        return func::logical_xor<U>;
    case Reduction::NONE:
    default:
        return nullptr;
    }
}

template <typename dataType, typename indicesType>
void scatterNdUpdate(const dataType* const inputData,
                     const indicesType* const indices,
                     const dataType* const updates,
                     dataType* const outBuf,
                     const Shape& dataShape,
                     const Shape& indicesShape,
                     const Shape& updatesShape,
                     const Reduction reduction_type = Reduction::NONE) {
    const auto update_chunk_shape = span(dataShape).drop_front(indicesShape.back());
    const auto update_el_number = shape_size(update_chunk_shape);

    std::memcpy(outBuf, inputData, sizeof(dataType) * shape_size(dataShape));

    const auto input_data_dim_pading = [&] {
        std::vector<size_t> padding(dataShape.size(), 1);
        for (size_t i = dataShape.size() - 1; i != 0; --i) {
            padding[i - 1] = padding[i] * dataShape[i];
        };
        return padding;
    }();
    const auto reduction = reduction_functor_for<dataType>(reduction_type);
    std::vector<indicesType> indicesCopy(indices, indices + shape_size(indicesShape));
    const auto num_of_updates = shape_size(span(indicesShape).drop_back(1));
    for (size_t i = 0; i != num_of_updates; ++i) {
        const auto indices_coord = indicesCopy.data() + i * indicesShape.back();
        const auto coord = span(indices_coord, indicesShape.back());

        // Negative value for indices means counting backwards from the end.
        int j = 0;
        for (auto& c : coord) {
            if (c < 0) {
                c += static_cast<indicesType>(dataShape[j]);
            }
            j++;
        }

        const auto out_index = std::inner_product(begin(coord), end(coord), begin(input_data_dim_pading), uint64_t(0));

        const auto update_data = updates + i * update_el_number;
        OPENVINO_ASSERT(out_index >= 0 && out_index + update_el_number <= shape_size(dataShape),
                        "Index is out of bounds");
        if (reduction) {
            std::transform(outBuf + out_index,
                           outBuf + out_index + update_el_number,
                           update_data,
                           outBuf + out_index,
                           reduction);
        } else {
            std::memcpy(outBuf + out_index, update_data, update_el_number * sizeof(dataType));
        }
    }
}
}  // namespace reference
}  // namespace ov
