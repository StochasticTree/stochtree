#ifndef STOCHTREE_OPENMP_UTILS_H
#define STOCHTREE_OPENMP_UTILS_H

#include <stochtree/common.h>
#include <algorithm>

namespace StochTree {

#ifdef _OPENMP

#include <omp.h>
#define STOCHTREE_HAS_OPENMP 1

// OpenMP thread management
inline int get_max_threads() {
    return omp_get_max_threads();
}

inline int get_thread_num() {
    return omp_get_thread_num();
}

inline int get_num_threads() {
    return omp_get_num_threads();
}

inline void set_num_threads(int num_threads) {
    omp_set_num_threads(num_threads);
}
    
#define STOCHTREE_PARALLEL_FOR(num_threads) \
    _Pragma("omp parallel for num_threads(num_threads)")

#define STOCHTREE_REDUCTION_ADD(var) \
    _Pragma("omp reduction(+:var)")

#define STOCHTREE_CRITICAL \
    _Pragma("omp critical")

#else
#define STOCHTREE_HAS_OPENMP 0

inline int get_max_threads() {return 1;}

inline int get_thread_num() {return 0;}

inline int get_num_threads() {return 1;}

inline void set_num_threads(int num_threads) {}
    
#define STOCHTREE_PARALLEL_FOR(num_threads)
    
#define STOCHTREE_REDUCTION_ADD(var)

#define STOCHTREE_CRITICAL

#endif

static int GetMaxThreads() {
    return get_max_threads();
}

static int GetCurrentThreadNum() {
    return get_thread_num();
}
    
static int GetNumThreads() {
    return get_num_threads();
}
    
static void SetNumThreads(int num_threads) {
    set_num_threads(num_threads);
}
    
static bool IsOpenMPAvailable() {
    return STOCHTREE_HAS_OPENMP;
}
    
static int GetOptimalThreadCount(int workload_size, int min_work_per_thread = 1000) {
    if (!IsOpenMPAvailable()) {
        return 1;
    }
    
    int max_threads = GetMaxThreads();
    int optimal_threads = workload_size / min_work_per_thread;
    
    return std::min(optimal_threads, max_threads);
}

// Parallel execution utilities
template<typename Func>
void ParallelFor(int start, int end, int num_threads, Func func) {
    if (num_threads <= 0) {
        num_threads = GetOptimalThreadCount(end - start);
    }
    
    if (num_threads == 1 || !STOCHTREE_HAS_OPENMP) {
        // Sequential execution
        for (int i = start; i < end; ++i) {
            func(i);
        }
    } else {
        // Parallel execution
        STOCHTREE_PARALLEL_FOR(num_threads)
        for (int i = start; i < end; ++i) {
            func(i);
        }
    }
}

} // namespace StochTree

#endif // STOCHTREE_OPENMP_UTILS_H 