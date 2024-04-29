// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"


namespace ov {
namespace snippets {
namespace lowered {

class SpecificIterationHandlers {
public:
    enum class HandlerType { FIRST_ITER, MAIN_BODY, LAST_ITER };
    SpecificIterationHandlers() = default;
    SpecificIterationHandlers(size_t loop_work_amount, size_t loop_increment);
    SpecificIterationHandlers(pass::PassPipeline first_iter_handlers,
                              pass::PassPipeline main_body_handlers,
                              pass::PassPipeline last_iter_handlers);

    const pass::PassPipeline& get_first_iter_handlers() const;
    const pass::PassPipeline& get_main_iter_handlers() const;
    const pass::PassPipeline& get_last_iter_handlers() const;
    static SpecificIterationHandlers merge_handlers(const SpecificIterationHandlers& lhs, const SpecificIterationHandlers& rhs);

    template <HandlerType Type,
              typename T,
              class... Args,
              typename std::enable_if<Type == HandlerType::FIRST_ITER, bool>::type = true>
    void register_handler(Args&&... args) {
        m_first_iter_handlers.register_pass<T>(args...);
    }

    template <HandlerType Type,
              typename T,
              class... Args,
              typename std::enable_if<Type == HandlerType::MAIN_BODY, bool>::type = true>
    void register_handler(Args&&... args) {
        m_main_body_handlers.register_pass<T>(args...);
    }

    template <HandlerType Type,
              typename T,
              class... Args,
              typename std::enable_if<Type == HandlerType::LAST_ITER, bool>::type = true>
    void register_handler(Args&&... args) {
        m_last_iter_handlers.register_pass<T>(args...);
    }

private:
    pass::PassPipeline m_first_iter_handlers;
    pass::PassPipeline m_main_body_handlers;
    pass::PassPipeline m_last_iter_handlers;
};

} // namespace lowered
} // namespace snippets
} // namespace ov
