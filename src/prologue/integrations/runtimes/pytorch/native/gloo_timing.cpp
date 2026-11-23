#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <time.h>
#include <unordered_map>
#include <unistd.h>

namespace {

struct GlooSample {
    std::uint64_t post_ns = 0;
    std::uint64_t completion_begin_ns = 0;
    std::uint64_t completion_end_ns = 0;
    std::uint64_t slot = 0;
    int peer = 0;
    int completion_thread = 0;
};

struct Endpoint {
    bool receiving;
    GlooSample sample;
};

struct Posting {
    std::uint64_t ticket = 0;
    bool receiving = false;
    const void * buffer = nullptr;
};

std::atomic_bool observing = false;
std::mutex mutex;
std::uint64_t next_ticket = 1;
std::unordered_map<const void *, std::uint64_t> pending;
std::unordered_map<std::uint64_t, Endpoint> endpoints;
thread_local Posting posting;

std::uint64_t timestamp() {
    timespec now {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) std::abort();
    return static_cast<std::uint64_t>(now.tv_sec) * 1'000'000'000 + static_cast<std::uint64_t>(now.tv_nsec);
}

template <typename Function>
Function native(const char * symbol) {

    // Python can load Torch locally. Resolve the definition from the loaded provider itself.
    static void * const library = dlopen("libtorch_cpu.so", RTLD_NOW | RTLD_NOLOAD);
    const auto target = library ? reinterpret_cast<Function>(dlsym(library, symbol)) : nullptr;
    if (!target) {
        std::fprintf(stderr, "Cannot resolve native Gloo symbol %s: %s\n", symbol, dlerror());
        std::abort();
    }
    return target;
}

void track(const void * buffer, bool receiving, int peer, std::uint64_t slot) {
    if (!posting.ticket) return;

    const std::lock_guard lock(mutex);
    if (!observing.load(std::memory_order_relaxed)) return;
    if (posting.buffer) {

        // One Python point-to-point call may claim only one native buffer.
        if (const auto old = pending.find(posting.buffer); old != pending.end() && old->second == posting.ticket)
            pending.erase(old);
        endpoints.erase(posting.ticket);
        posting = {};
        return;
    }
    if (posting.receiving != receiving || !pending.emplace(buffer, posting.ticket).second) {
        posting = {};
        return;
    }

    posting.buffer = buffer;
    endpoints.emplace(posting.ticket, Endpoint{ receiving, { timestamp(), 0, 0, slot, peer, 0 } });
}

using NativePost = void (*)(void *, int, std::uint64_t, std::size_t, std::size_t);
using NativeCompletion = void (*)(void *, int);

void complete(NativeCompletion original, void * buffer, bool receiving, int peer) {
    std::uint64_t ticket = 0;
    if (observing.load(std::memory_order_relaxed)) {
        const std::lock_guard lock(mutex);
        if (const auto match = pending.find(buffer); match != pending.end()) {
            const auto & endpoint = endpoints.at(match->second);
            if (endpoint.receiving == receiving && endpoint.sample.peer == peer) {
                ticket = match->second;
                pending.erase(match);
            }
        }
    }

    if (!ticket) {
        original(buffer, peer);
        return;
    }

    // The real callback publishes readiness and may wake a consumer before it returns.
    const auto begin = timestamp();
    original(buffer, peer);
    const auto end = timestamp();
    const auto thread = gettid();
    const std::lock_guard lock(mutex);
    if (const auto row = endpoints.find(ticket); row != endpoints.end()) {
        row->second.sample.completion_begin_ns = begin;
        row->second.sample.completion_end_ns = end;
        row->second.sample.completion_thread = thread;
    }
}

} // namespace

extern "C" void prologue_gloo_begin_window() {
    const std::lock_guard lock(mutex);
    pending.clear();
    endpoints.clear();
    observing.store(true, std::memory_order_relaxed);
}

extern "C" std::uint64_t prologue_gloo_begin_post(int receiving) {
    const std::lock_guard lock(mutex);
    posting = observing.load(std::memory_order_relaxed) ? Posting{ next_ticket++, receiving != 0, nullptr } : Posting{};
    return posting.ticket;
}

extern "C" void prologue_gloo_end_post() { posting = {}; }

extern "C" void prologue_gloo_capture(const std::uint64_t * tickets, std::size_t count, GlooSample * output) {
    const std::lock_guard lock(mutex);
    observing.store(false, std::memory_order_relaxed);
    for (std::size_t index = 0; index < count; ++index) {
        const auto found = endpoints.find(tickets[index]);
        output[index] = found == endpoints.end() ? GlooSample{} : found->second.sample;
    }
    pending.clear();
    endpoints.clear();
}

// Interpose the pinned TCP transport ABI without importing Torch types into the adapter.
extern "C" void gloo_send(void *, int, std::uint64_t, std::size_t, std::size_t)
    asm("_ZN4gloo9transport3tcp13UnboundBuffer4sendEimmm");
extern "C" void gloo_receive(void *, int, std::uint64_t, std::size_t, std::size_t)
    asm("_ZN4gloo9transport3tcp13UnboundBuffer4recvEimmm");
extern "C" void gloo_send_ready(void *, int) asm("_ZN4gloo9transport3tcp13UnboundBuffer20handleSendCompletionEi");
extern "C" void gloo_receive_ready(void *, int) asm("_ZN4gloo9transport3tcp13UnboundBuffer20handleRecvCompletionEi");

void gloo_send(void * buffer, int peer, std::uint64_t slot, std::size_t offset, std::size_t bytes) {
    static const auto original = native<NativePost>("_ZN4gloo9transport3tcp13UnboundBuffer4sendEimmm");
    track(buffer, false, peer, slot);
    original(buffer, peer, slot, offset, bytes);
}

void gloo_receive(void * buffer, int peer, std::uint64_t slot, std::size_t offset, std::size_t bytes) {
    static const auto original = native<NativePost>("_ZN4gloo9transport3tcp13UnboundBuffer4recvEimmm");
    track(buffer, true, peer, slot);
    original(buffer, peer, slot, offset, bytes);
}

void gloo_send_ready(void * buffer, int peer) {
    static const auto original = native<NativeCompletion>("_ZN4gloo9transport3tcp13UnboundBuffer20handleSendCompletionEi");
    complete(original, buffer, false, peer);
}

void gloo_receive_ready(void * buffer, int peer) {
    static const auto original = native<NativeCompletion>("_ZN4gloo9transport3tcp13UnboundBuffer20handleRecvCompletionEi");
    complete(original, buffer, true, peer);
}
