// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/general_utils.h"

namespace CPUTestUtils {

std::vector<CPUSpecificParams> filterCPUInfo(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> archCPUParams = filterCPUInfoForArch(CPUParams);
    std::vector<CPUSpecificParams> deviceCPUParams = filterCPUInfoForDevice(archCPUParams);
    return deviceCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForArch(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("acl") != std::string::npos)
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;
    const int inputFormatIndex = 0;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);
        auto inputsFormat = std::get<inputFormatIndex>(param);
        if (!inputsFormat.empty() && !selectedTypeStr.empty() && selectedTypeStr == "any_type") {
            if (ov::intel_cpu::one_of(inputsFormat[0], nCw8c, nChw8c, nCdhw8c) && !ov::with_cpu_x86_sse42())
                continue;
            if (ov::intel_cpu::one_of(inputsFormat[0], nCw16c, nChw16c, nCdhw16c) && !ov::with_cpu_x86_avx512f())
                continue;
        }
        if (selectedTypeStr.find("jit") != std::string::npos && !ov::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !ov::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx") != std::string::npos && !ov::with_cpu_x86_avx())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !ov::with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !ov::with_cpu_x86_avx512f())
            continue;
        if (selectedTypeStr.find("amx") != std::string::npos && !ov::with_cpu_x86_avx512_core_amx())
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}

} // namespace CPUTestUtils
