// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

#include <iostream>

static void update(const std::unordered_map<std::shared_ptr<ov::Symbol>, std::shared_ptr<ov::Symbol>>& original,
                   std::unordered_map<std::shared_ptr<ov::Symbol>, std::shared_ptr<ov::Symbol>>& result) {
    for (const auto& item : original) {
        const auto& B_root = ov::symbol::ancestor_of(item.first);
        const auto& C_root = ov::symbol::ancestor_of(item.second);
        if (result.count(B_root)) {
            // we've found two equations A + B = C and A + B = D; we deduced that C == D;
            ov::symbol::set_equal(result[B_root], C_root);
        } else {  // key B is proved to be unique so far, check for uniqueness of C
            bool record_already_in_result = false;
            for (const auto& item_i : result) {
                if (ov::symbol::are_equal(item_i.second, C_root)) {
                    // we've found two equations K + N = M and K + L = M; we deduced N == L;
                    ov::symbol::set_equal(item_i.first, B_root);
                    record_already_in_result = true;
                }  // we continue to search for equal results of summation to cover all of them
            }
            if (!record_already_in_result)
                result[B_root] = C_root;
        }
    }
}

static std::unordered_map<std::shared_ptr<ov::Symbol>, std::shared_ptr<ov::Symbol>> merge_and_normalize(
    const std::unordered_map<std::shared_ptr<ov::Symbol>, std::shared_ptr<ov::Symbol>>& lhs,
    const std::unordered_map<std::shared_ptr<ov::Symbol>, std::shared_ptr<ov::Symbol>>& rhs) {
    std::unordered_map<std::shared_ptr<ov::Symbol>, std::shared_ptr<ov::Symbol>> result;
    //    if (!lhs.empty() || !rhs.empty())
    //        std::cout << "Symbol update" << std::endl;
    update(lhs, result);
    update(rhs, result);
    return result;
}

std::shared_ptr<ov::Symbol> ov::symbol::ancestor_of(const std::shared_ptr<Symbol>& symbol) {
    auto x = symbol;
    while (x->m_parent) {
        if (x->m_parent->m_parent)
            x->m_parent = x->m_parent->m_parent;
        x = x->m_parent;
    }
    return x;
}

bool ov::symbol::are_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return false;
    return ov::symbol::ancestor_of(lhs).get() == ov::symbol::ancestor_of(rhs).get();
}

void ov::symbol::set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return;
    auto lhs_root = ov::symbol::ancestor_of(lhs), rhs_root = ov::symbol::ancestor_of(rhs);
    if (lhs_root.get() == rhs_root.get())
        return;  // already are equal
    lhs_root->m_parent = rhs_root;
    rhs_root->m_sum = merge_and_normalize(rhs_root->m_sum, lhs_root->m_sum);
    lhs_root->m_sum = {};
}

std::shared_ptr<ov::Symbol> ov::operator+(const std::shared_ptr<ov::Symbol>& lhs,
                                          const std::shared_ptr<ov::Symbol>& rhs) noexcept {
    if (!lhs || !rhs)
        return nullptr;
    const auto& lhs_root = ov::symbol::ancestor_of(lhs);
    const auto& rhs_root = ov::symbol::ancestor_of(rhs);
    // if rhs has record of being summed with lhs -- than return resulting symbol
    for (const auto& item : lhs_root->m_sum)
        if (ov::symbol::are_equal(ov::symbol::ancestor_of(item.first), rhs_root))
            return item.second;
    auto new_symbol = std::make_shared<ov::Symbol>();
    lhs_root->m_sum[rhs_root] = new_symbol;
    rhs_root->m_sum[lhs_root] = new_symbol;
    return std::make_shared<ov::Symbol>();
}
