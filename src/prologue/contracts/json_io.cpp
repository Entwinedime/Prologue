#include "prologue/contracts/json_io.hpp"

#include <atomic>
#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>
#include <unistd.h>

namespace prologue {

nlohmann::json read_json(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read input: " + path.string());
    input.exceptions(std::ios::badbit);
    return nlohmann::json::parse(input);
}

Publication::Publication(std::filesystem::path destination) : destination_(std::move(destination)) {
    if (!destination_.parent_path().empty()) std::filesystem::create_directories(destination_.parent_path());
    static std::atomic<unsigned long> sequence{};

    // Exclusive creation also handles temporary files left by a previous process with the same PID.
    do {
        temporary_ = destination_.string() + ".tmp." + std::to_string(getpid()) + "." + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
        stream_.open(temporary_, std::ios::out | std::ios::noreplace);
    } while (!stream_ && errno == EEXIST);
    stream_.exceptions(std::ios::badbit | std::ios::failbit);
}

Publication::~Publication() {
    std::error_code ignored;
    std::filesystem::remove(temporary_, ignored);
}

void Publication::append(const nlohmann::json & record) { stream_ << record << '\n'; }

void Publication::write(const nlohmann::json & last) {
    append(last);
    stream_.close();

    // A same-directory hard link publishes the closed file without replacing a prior destination.
    std::filesystem::create_hard_link(temporary_, destination_);
}

} // namespace prologue
