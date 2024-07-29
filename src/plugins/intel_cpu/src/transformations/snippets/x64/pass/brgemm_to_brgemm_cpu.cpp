// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "brgemm_to_brgemm_cpu.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/tpp/x64/op/modifiers.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#include "cpu_shape.h"
#include "utils/general_utils.h"


namespace ov {
namespace intel_cpu {

using namespace snippets::lowered;

namespace {
std::vector<size_t> make_subtensor(const ov::Shape& tensor) {
    return std::vector<size_t>(std::min(tensor.size(), size_t(2)), ov::snippets::utils::get_full_dim_value());
}
template<typename T>
void set_full_port_desc(const T& port) {
    const auto& shape = port.get_shape();
    PortDescriptorUtils::set_port_descriptor_ptr(port, std::make_shared<PortDescriptor>(shape, make_subtensor(shape)));
}
template<typename T, typename... Args>
void set_port_desc(const T& port, Args... params) {
    PortDescriptorUtils::set_port_descriptor_ptr(port, std::make_shared<PortDescriptor>(params...));
}
} // namespace

pass::BrgemmToBrgemmCPU::BrgemmToBrgemmCPU() {
    MATCHER_SCOPE(BrgemmToBrgemmCPU);
    auto is_not_tpp = [](const Output<Node>& out) {
        return !std::dynamic_pointer_cast<const intel_cpu::tpp::modifier::TensorProcessingPrimitive>(out.get_node_shared_ptr());
    };
    auto m_brgemm = ov::pass::pattern::wrap_type<snippets::op::Brgemm>(is_not_tpp);

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::BrgemmToBrgemmCPU")
        const auto node = m.get_match_root();
        const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(node);
        const auto brgemm_plugin = ov::as_type_ptr<BrgemmCPU>(node);
        if (!brgemm || brgemm_plugin)
            OPENVINO_THROW("BrgemmCPU cannot be in body before BrgemmToBrgemmCPU pass");

        const auto& brgemm_in0_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0));
        const auto& brgemm_in1_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1));
        const auto& brgemm_out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));

        const auto dimsMatMulIn0 = snippets::utils::get_planar_pshape(brgemm->input(0));
        const auto dimsMatMulIn1 = snippets::utils::get_planar_pshape(brgemm->input(1));

        const auto K = *dimsMatMulIn0.rbegin();
        const auto N = *dimsMatMulIn1.rbegin();

        const auto element_type_a = brgemm->get_input_element_type(0);
        auto brgemm_type = brgemm_utils::get_brgemm_type(element_type_a, K, N);
        const auto offset_a = brgemm->get_offset_a();
        const auto offset_b = brgemm->get_offset_b();
        const auto offset_c = brgemm->get_offset_c();

        std::shared_ptr<BrgemmCPU> brgemm_cpu = nullptr;
        std::shared_ptr<BrgemmCopyB> brgemm_repacking = nullptr;
        if (stand_alone(brgemm_type)) {
            brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0), brgemm->input_value(1), brgemm_type,
                                                     offset_a, offset_b, offset_c,
                                                     brgemm_in0_desc->get_layout(), brgemm_in1_desc->get_layout(), brgemm_out_desc->get_layout());
        } else {
            const auto copy_b_type = with_compensations(brgemm_type) ? brgemm_type : brgemm_utils::BRGEMM_TYPE::REPACKING_ONLY;
            brgemm_repacking = std::make_shared<BrgemmCopyB>(brgemm->input_value(1), element_type_a, copy_b_type, offset_b, 0, 0,
                                                             brgemm_in1_desc->get_layout());
            set_port_desc(brgemm_repacking->input(0), brgemm_in1_desc->get_shape(), brgemm_in1_desc->get_subtensor(), brgemm_in1_desc->get_layout());
            set_full_port_desc(brgemm_repacking->output(0));

            if (with_amx(brgemm_type)) {
                const auto scratch = std::make_shared<snippets::op::NewMemoryBuffer>(ov::Shape{BrgemmCPU::SCRATCH_BYTE_SIZE});
                brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0), brgemm_repacking->output(0), scratch, brgemm_type,
                                                         offset_a, offset_b, 0, offset_c,
                                                         brgemm_in0_desc->get_layout(), std::vector<size_t>{}, brgemm_out_desc->get_layout());
                set_full_port_desc(scratch->output(0));
                set_full_port_desc(brgemm_cpu->input(2));
            } else if (with_compensations(brgemm_type)) {
                brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0), brgemm_repacking->output(0), brgemm_repacking->output(1),
                                                         brgemm_type, offset_a, offset_b, 0, offset_c,
                                                         brgemm_in0_desc->get_layout(), std::vector<size_t>{}, brgemm_out_desc->get_layout());
                set_full_port_desc(brgemm_repacking->output(1));
                set_full_port_desc(brgemm_cpu->input(2));
            } else if (repacking_only(brgemm_type)) {
                brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0), brgemm_repacking->output(0), brgemm_type,
                                                         offset_a, offset_b, offset_c,
                                                         brgemm_in0_desc->get_layout(), std::vector<size_t>{}, brgemm_out_desc->get_layout());
            } else {
                OPENVINO_THROW("Invalid configuration for BRGEMM CPU");
            }
        }

        brgemm_cpu->set_friendly_name(brgemm->get_friendly_name());
        ov::replace_node(brgemm, brgemm_cpu);

        // Transfer ports
        set_port_desc(brgemm_cpu->input(0), brgemm_in0_desc->get_shape(), brgemm_in0_desc->get_subtensor(), brgemm_in0_desc->get_layout());
        if (brgemm_repacking) {
            set_full_port_desc(brgemm_cpu->input(1));
        } else {
            set_port_desc(brgemm_cpu->input(1), brgemm_in1_desc->get_shape(), brgemm_in1_desc->get_subtensor(), brgemm_in1_desc->get_layout());
        }
        set_port_desc(brgemm_cpu->output(0), brgemm_out_desc->get_shape(), brgemm_out_desc->get_subtensor(), brgemm_out_desc->get_layout());

        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())
        if (brgemm_repacking)
            brgemm_repacking->validate_and_infer_types();
        brgemm_cpu->validate_and_infer_types();

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_brgemm, matcher_name);
    register_matcher(m, callback);
}
} // namespace intel_cpu
} // namespace ov
