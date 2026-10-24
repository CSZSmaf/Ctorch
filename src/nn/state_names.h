#pragma once

#include "dl/nn.h"

#include <string_view>
#include <unordered_map>

namespace dl::nn::state_detail {

// 旧状态中的共享路径可映射到当前规范名；显式提供的规范项保持优先。
template <class Value>
void add_aliases(std::unordered_map<std::string_view, Value>& values, std::span<const NameAlias> aliases) {
    for (const auto& [alias, canonical] : aliases) {
        if (auto source = values.find(alias); source != values.end()) {
            values.try_emplace(canonical, source->second);
        }
    }
}

inline std::vector<std::string> canonical_names(std::span<const std::string> names,
                                               std::span<const NameAlias> aliases) {
    std::unordered_map<std::string_view, std::string_view> lookup;
    for (const auto& [alias, canonical] : aliases) lookup.emplace(alias, canonical);
    std::vector<std::string> result(names.begin(), names.end());
    for (auto& name : result) {
        if (auto entry = lookup.find(name); entry != lookup.end()) name = entry->second;
    }
    return result;
}

} // namespace dl::nn::state_detail
