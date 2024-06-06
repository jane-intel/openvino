// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/cancellation.hpp"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "overload/ov_infer_request/cancellation.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCancellationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCancellationTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

}  // namespace
