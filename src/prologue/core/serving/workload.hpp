#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace prologue::serving {

struct Request final {
    std::string id;
    std::int64_t arrival_ns;
    std::int64_t prompt_tokens;
    std::int64_t output_tokens;
};

class Workload final {
public:
    Workload(std::string id, std::vector<Request> requests);

    const std::string& id() const;
    const std::vector<Request>& requests() const;

private:
    std::string id_;
    std::vector<Request> requests_;
};

}
