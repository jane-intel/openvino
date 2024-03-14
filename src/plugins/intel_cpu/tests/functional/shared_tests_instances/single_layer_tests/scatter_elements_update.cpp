// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/scatter_elements_update.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ScatterElementsUpdateLayerTest;

namespace {
// map<input_shape, map<indices_shape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape {
    {{10, 12, 15}, {{{1, 2, 4}, {0, 1, 2}}, {{2, 2, 2}, {-1, -2, -3}}}},
    {{15, 9, 8, 12}, {{{1, 2, 2, 2}, {0, 1, 2, 3}}, {{1, 2, 1, 4}, {-1, -2, -3, -4}}}},
    {{9, 9, 8, 8, 11, 10}, {{{1, 2, 1, 2, 1, 2}, {5, -3}}}},
};
// index value should not be random data
const std::vector<std::vector<size_t>> idx_value = {
        {1, 0, 4, 6, 2, 3, 7, 5}
};

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::element::Type> idx_types = {
        ov::element::i32,
        ov::element::i64,
};

std::vector<ov::test::axisShapeInShape> combine_shapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::axisShapeInShape> res_vec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            for (auto& elt : item.second) {
                res_vec.push_back(ov::test::axisShapeInShape{
                    ov::test::static_shapes_to_test_representation({input_shape.first, item.first}),
                    elt});
            }
        }
    }
    return res_vec;
}

const auto scatter_elt_update_cases = ::testing::Combine(
        ::testing::ValuesIn(combine_shapes(axesShapeInShape)),
        ::testing::ValuesIn(idx_value),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(idx_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterEltsUpdate, ScatterElementsUpdateLayerTest,
    scatter_elt_update_cases, ScatterElementsUpdateLayerTest::getTestCaseName);

}  // namespace
