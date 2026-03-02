#include "hwy/highway.h"
namespace hn = hwy::HWY_NAMESPACE;
int main() {
    hn::ScalableTag<float> d;
    auto v = hn::Zero(d);
    hn::StoreU(v, d, (float*)0);
    return 0;
}
