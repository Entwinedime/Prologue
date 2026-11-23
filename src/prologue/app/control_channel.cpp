#include "prologue/app/control_channel.hpp"

#include "prologue/core/performance/costs.hpp"
#include "prologue/core/serving/simulation.hpp"
#include "prologue/core/structure/templates/library.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace prologue::app {
namespace {

using nlohmann::json;

class Socket final {
public:
    explicit Socket(int descriptor) : descriptor_(descriptor) {
        check(descriptor_, "open control socket");
    }

    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    ~Socket() {
        close(descriptor_);
    }

    Socket accept_at(const std::string& path) const {
        sockaddr_un address{};
        if (path.size() >= sizeof(address.sun_path)) {
            throw std::invalid_argument("control socket path is too long");
        }
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        check(bind(descriptor_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), "bind control socket");
        try {
            check(listen(descriptor_, 1), "listen for native control");
            const auto peer = accept(descriptor_, nullptr, nullptr);
            check(peer, "accept native control");
            unlink(path.c_str());
            return Socket(peer);
        } catch (...) {
            unlink(path.c_str());
            throw;
        }
    }

    std::optional<json> receive() {
        std::array<char, 4096> bytes{};
        auto end = pending_.find('\n');
        while (end == std::string::npos) {
            const auto size = recv(descriptor_, bytes.data(), bytes.size(), 0);
            check(size, "read control message");
            if (size == 0) {
                return std::nullopt;
            }
            pending_.append(bytes.data(), static_cast<std::size_t>(size));
            end = pending_.find('\n');
        }
        auto message = json::parse(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(end));
        pending_.erase(0, end + 1);
        return message;
    }

    void respond(const json& value) const {
        const auto message = value.dump() + '\n';
        std::string_view bytes(message);
        while (!bytes.empty()) {
            const auto sent = send(descriptor_, bytes.data(), bytes.size(), MSG_NOSIGNAL);
            check(sent, "write control message");
            bytes.remove_prefix(static_cast<std::size_t>(sent));
        }
    }

private:
    static void check(long result, const char* operation) {
        if (result < 0) {
            throw std::system_error(errno, std::generic_category(), operation);
        }
    }

    int descriptor_;
    std::string pending_;
};

}

nlohmann::json serve_control(const std::string& socket_path, serving::Simulation& simulation) {
    auto channel = Socket(socket(AF_UNIX, SOCK_STREAM, 0)).accept_at(socket_path);
    while (const auto message = channel.receive()) {
        try {
            auto reply = simulation.request(*message);
            const bool drained = message->at("operation") == "arrivals" && reply.at("done").get<bool>();
            channel.respond({{"now_ns", simulation.now()}, {"result", std::move(reply)}});
            if (drained) {
                return simulation.result("complete", "");
            }
        } catch (const serving::Stopped& stopped) {
            channel.respond({{"failure", stopped.what()}});
            return simulation.result(stopped.status(), stopped.what());
        } catch (const structure::TemplateGap& gap) {
            channel.respond({{"failure", gap.what()}});
            auto result = simulation.result(gap.detail.at("status"), gap.what());
            result["gap"] = gap.detail;
            return result;
        } catch (const performance::CostGap& gap) {
            channel.respond({{"failure", gap.what()}});
            auto result = simulation.result(gap.detail.at("status"), gap.what());
            result["gap"] = gap.detail;
            return result;
        }
    }
    return simulation.result("controller_error", "native controller closed before completion");
}

}
