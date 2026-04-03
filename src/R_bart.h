#pragma once
#include <stochtree/bart.h>
#include <memory>

// Wrapper used by cpp11 external_pointer to manage a BARTResult heap object.
struct BARTResultR {
    std::unique_ptr<StochTree::BARTResult> result;
    explicit BARTResultR(std::unique_ptr<StochTree::BARTResult> r)
        : result(std::move(r)) {}
};
