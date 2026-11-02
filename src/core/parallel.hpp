#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace splat {

// How many threads `threads` means: 0 is every core.
[[nodiscard]] inline unsigned thread_count(int threads) {
  return threads > 0 ? static_cast<unsigned>(threads) : std::max(1u, std::thread::hardware_concurrency());
}

// Runs job(i) for i in [0, count) on up to `threads` threads, handing out
// indices one at a time, and returns when all are done.
template <class Job> void parallel_for(std::size_t count, int threads, const Job &job) {
  const unsigned wanted = thread_count(threads);
  std::atomic<std::size_t> next{0};
  std::vector<std::jthread> workers;
  for (unsigned t = 0; t < std::min<std::size_t>(wanted, count); ++t) {
    workers.emplace_back([&] {
      for (std::size_t i = next++; i < count; i = next++) job(i);
    });
  }
}

// Like parallel_for, but each thread gets its own index, so the job can keep
// per-thread state: job(thread, i) with thread in [0, threads_used).
template <class Job> void parallel_for_threaded(std::size_t count, unsigned threads, const Job &job) {
  threads = static_cast<unsigned>(std::min<std::size_t>(std::max(threads, 1u), std::max<std::size_t>(count, 1)));
  if (threads == 1) {
    for (std::size_t i = 0; i < count; ++i) job(0u, i);
    return;
  }
  std::atomic<std::size_t> next{0};
  std::vector<std::jthread> workers;
  for (unsigned t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      for (std::size_t i = next++; i < count; i = next++) job(t, i);
    });
  }
}

} // namespace splat
