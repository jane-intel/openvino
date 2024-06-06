// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "lir_comparator.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace test {
namespace snippets {
class LoweredPassTestsF : public ov::test::TestsCommon {
public:
    LoweredPassTestsF();

    void SetUp() override {}

    void TearDown() override;

    std::shared_ptr<ov::snippets::lowered::LinearIR> linear_ir, linear_ir_ref;
    ov::snippets::lowered::pass::PassPipeline pipeline;
    LIRComparator comparator;
};

/**
 * @brief Returns default 2D subtensor filled with FULL_DIM values.
 * @return default subtensor
 */
ov::snippets::VectorDims get_default_subtensor();

/**
 * @brief Inits input and output descriptors, and sets them to expression and its ov::Node.
 * @attention Descriptor shapes are initialized using ov::Node input/output shapes
 * @attention If optional vector of parameters (subtensors or layouts) is set, its size must be equal to n_inputs + n_outputs
 * @attention If subtensors are not set, default 2D subtensor (filled with FULL_DIM values) is created
 * @param expr expression whose descriptors should be initialized
 * @param subtensors vector of subtensors to set
 * @param layouts vector of layouts to set
 */
void init_expr_descriptors(const ov::snippets::lowered::ExpressionPtr& expr,
                           const std::vector<ov::snippets::VectorDims>& subtensors = {},
                           const std::vector<ov::snippets::VectorDims>& layouts = {});

/**
 * @brief Creates unified loop info based on provided entry and exit points, and adds it to the linear_ir's loops map
 * @attention This helper wraps LoopManager::mark_loop method, but only for LoopInfo creation (whereas original
 * mark_loop method also marks expressions with the corresponding loop info).
 * @param linear_ir linear_ir in which loop info should be added
 * @param entries entry points of loop
 * @param exits exit points of loop
 */
void create_and_add_unified_loop_info(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                                      size_t work_amount,
                                      size_t increment,
                                      const std::vector<ov::snippets::lowered::LoopPort>& entries,
                                      const std::vector<ov::snippets::lowered::LoopPort>& exits,
                                      bool add_default_handlers = true);
}  // namespace snippets
}  // namespace test
}  // namespace ov
