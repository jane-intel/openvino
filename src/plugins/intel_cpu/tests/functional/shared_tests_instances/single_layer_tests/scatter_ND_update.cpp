// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/scatter_ND_update.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ScatterNDUpdateLayerTest;

namespace {
const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::element::Type> idx_types = {
        ov::element::i32,
        ov::element::i64,
};
// map<input_shape map<indices_shape, indices_value>>
// update_shape is gotten from input_shape and indices_shape
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> sliceSelectInShape {
    {{10, 9, 9, 11}, {{{4, 1}, {1, 3, 5, 7}}, {{1, 2}, {4, 6}}, {{2, 3}, {0, 1, 1, 2, 2, 2}}, {{1, 4}, {5, 5, 4, 9}}}},
    {{10, 9, 10, 9, 10}, {{{2, 2, 1}, {5, 6, 2, 8}}, {{2, 3}, {0, 4, 6, 5, 7, 1}}}},
};

std::vector<ov::test::scatterNDUpdateSpecParams> combineShapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>>& input_shapes) {
    std::vector<ov::test::scatterNDUpdateSpecParams> resVec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            auto indices_shape = item.first;
            size_t indices_rank = indices_shape.size();
            std::vector<size_t> update_shape;
            for (size_t i = 0; i < indices_rank - 1; i++) {
                update_shape.push_back(indices_shape[i]);
            }
            auto src_shape = input_shape.first;
            for (size_t j = indices_shape[indices_rank - 1]; j < src_shape.size(); j++) {
                update_shape.push_back(src_shape[j]);
            }
            std::vector<ov::Shape> in_shapes{src_shape, update_shape};
            resVec.push_back(
                ov::test::scatterNDUpdateSpecParams{
                                ov::test::static_shapes_to_test_representation(in_shapes),
                                ov::Shape{indices_shape},
                                item.second});
        }
    }
    return resVec;
}

const auto ScatterNDUpdateCases = ::testing::Combine(
        ::testing::ValuesIn(combineShapes(sliceSelectInShape)),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(idx_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate, ScatterNDUpdateLayerTest, ScatterNDUpdateCases, ScatterNDUpdateLayerTest::getTestCaseName);

}  // namespace
