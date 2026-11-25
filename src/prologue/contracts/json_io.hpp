#pragma once

#include <filesystem>
#include <fstream>

#include <nlohmann/json_fwd.hpp>

namespace prologue {

nlohmann::json read_json(const std::filesystem::path & path);

class Publication {
public:
    explicit Publication(std::filesystem::path destination);
    ~Publication();

    void append(const nlohmann::json & record);
    void write(const nlohmann::json & last);

private:
    std::filesystem::path destination_;
    std::filesystem::path temporary_;
    std::ofstream stream_;
};

} // namespace prologue
