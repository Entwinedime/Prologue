#include "runtime_timing.hpp"

#include <cstdio>
#include <ctime>
#include <fstream>
#include <mutex>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

namespace prologue::ascend {

std::atomic<Observation> observation{ Observation::Off };

namespace {

struct Interval {
    const char * api;
    long thread;
    int result;
    std::uint32_t tasks;
    std::size_t bytes;
    int direction;
    std::uintptr_t source;
    std::int64_t before_ns;
    std::int64_t begin_ns;
    std::int64_t end_ns;
    std::int64_t after_ns;
};

struct QueueMark {
    long thread;
    std::int64_t counter;
    std::int64_t recorded_ns;
};

std::mutex samples_mutex;
std::vector<Interval> samples;
std::vector<QueueMark> queue_marks;

std::int64_t now_ns() {
    timespec time{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);
    return static_cast<std::int64_t>(time.tv_sec) * 1'000'000'000 + static_cast<std::int64_t>(time.tv_nsec);
}

} // namespace

RuntimeSample::RuntimeSample(const char * name, bool enabled)
    : name_(name), before_(enabled && observation.load(std::memory_order_relaxed) == Observation::Timing ? now_ns() : 0) {}

void RuntimeSample::enter() {
    if (before_) begin_ = now_ns();
}

void RuntimeSample::finish(int result, std::uint32_t tasks, std::size_t bytes, int direction, const void * source) const {
    if (!begin_) return;
    const auto end = now_ns();
    thread_local const auto thread = syscall(SYS_gettid);

    // Keep observer work outside the actual API interval and defer all formatting until collection ends.
    std::lock_guard lock(samples_mutex);
    samples.push_back({ name_, thread, result, tasks, bytes, direction, reinterpret_cast<std::uintptr_t>(source),
                        before_, begin_, end, 0 });
    samples.back().after_ns = now_ns();
}

extern "C" bool prologue_ascend_host_timing_active() {
    return observation.load(std::memory_order_relaxed) == Observation::Timing;
}

extern "C" void prologue_ascend_queue_mark_recorded(std::int64_t counter) {
    if (observation.load(std::memory_order_relaxed) != Observation::Timing) return;

    thread_local const auto thread = syscall(SYS_gettid);
    std::lock_guard lock(samples_mutex);
    if (observation.load(std::memory_order_relaxed) != Observation::Timing) return;

    queue_marks.push_back({ thread, counter, 0 });
    queue_marks.back().recorded_ns = now_ns();
}

extern "C" void prologue_runtime_timing_start() {
    observation.store(Observation::Timing, std::memory_order_relaxed);
}

extern "C" int prologue_runtime_timing_stop(const char * path, const char * mark_path) {
    observation.store(Observation::Off, std::memory_order_relaxed);
    try {
        std::lock_guard lock(samples_mutex);
        std::ofstream output(path);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        for (const auto & sample : samples) {
            output << "{\"api\":\"" << sample.api << "\",\"thread\":" << sample.thread
                   << ",\"result\":" << sample.result << ",\"tasks\":" << sample.tasks
                   << ",\"bytes\":" << sample.bytes << ",\"direction\":" << sample.direction;
            if (sample.source) output << ",\"source\":\"" << std::hex << sample.source << std::dec << "\"";
            output << ",\"before_raw_ns\":" << sample.before_ns << ",\"begin_raw_ns\":" << sample.begin_ns
                   << ",\"end_raw_ns\":" << sample.end_ns << ",\"after_raw_ns\":" << sample.after_ns << "}\n";
        }
        output.close();

        std::ofstream marks(mark_path);
        marks.exceptions(std::ios::failbit | std::ios::badbit);
        for (const auto & mark : queue_marks) {
            marks << '[' << mark.thread << ',' << mark.counter << ',' << mark.recorded_ns << "]\n";
        }
        marks.close();

        samples.clear();
        queue_marks.clear();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "prologue: %s\n", error.what());
        return -1;
    }
}

} // namespace prologue::ascend
