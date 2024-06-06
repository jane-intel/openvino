// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/remote.hpp"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

std::vector<std::pair<ov::AnyMap, ov::AnyMap>> generate_remote_params() {
    return {{}};
}

// [Tracking number: E#110088]
// NPU plugin does not support `Remote Tensors` yet
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_BehaviorTests, OVRemoteTest,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::Values(::ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(generate_remote_params())),
                         (ov::test::utils::appendPlatformTypeTestName<OVRemoteTest>));
}  // namespace
