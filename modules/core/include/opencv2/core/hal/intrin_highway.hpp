// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Google Highway SIMD backend for OpenCV Universal Intrinsics (CV_SIMD_SCALABLE mode).
// Implements the same public interface as intrin_rvv_scalable.hpp but
// using Highway API so it can target multiple ISAs (SSE/AVX/NEON/RVV/SVE …).

#ifndef OPENCV_HAL_INTRIN_HIGHWAY_HPP
#define OPENCV_HAL_INTRIN_HIGHWAY_HPP

// ── Highway ──────────────────────────────────────────────────────────────────
// Must be included before any OpenCV namespace so that HWY_NAMESPACE is resolved
// at the file level.  We use HWY_STATIC_DISPATCH (no foreach_target machinery).
#ifndef __SPIRV_DEVICE__
#define __SPIRV_DEVICE__ 0
#endif

#include "hwy/highway.h"

#include <cstdint>
#include <cstring>   // memcpy
#include <algorithm> // std::min
#include <array>

namespace cv {

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

// ─────────────────────────────────────────────────────────────────────────────
// Scalable-SIMD feature flags
// ─────────────────────────────────────────────────────────────────────────────
#define CV_SIMD_SCALABLE    1
#define CV_SIMD_SCALABLE_64F 1

// Convenience alias for the runtime-selected Highway namespace
namespace hn = hwy::HWY_NAMESPACE;

// ─────────────────────────────────────────────────────────────────────────────
// Tag-type aliases  (zero-size descriptors that carry element type + ISA info)
// Use CappedTag to guarantee exactly 128-bit vectors on all architectures,
// including RISC-V RVV with VLEN > 128. This matches the cpu_baseline semantics
// where operations are defined for 128-bit SIMD width.
// ─────────────────────────────────────────────────────────────────────────────
using hwy_d_u8  = hn::ScalableTag<uint8_t>;
using hwy_d_s8  = hn::ScalableTag<int8_t>;
using hwy_d_u16 = hn::ScalableTag<uint16_t>;
using hwy_d_s16 = hn::ScalableTag<int16_t>;
using hwy_d_u32 = hn::ScalableTag<uint32_t>;
using hwy_d_s32 = hn::ScalableTag<int32_t>;
using hwy_d_u64 = hn::ScalableTag<uint64_t>;
using hwy_d_s64 = hn::ScalableTag<int64_t>;
using hwy_d_f32 = hn::ScalableTag<float>;
using hwy_d_f64 = hn::ScalableTag<double>;

// ─────────────────────────────────────────────────────────────────────────────
// OpenCV vector-type aliases  (Highway Vec<D> for every lane type)
// ─────────────────────────────────────────────────────────────────────────────
using v_uint8   = hn::Vec<hwy_d_u8>;
using v_int8    = hn::Vec<hwy_d_s8>;
using v_uint16  = hn::Vec<hwy_d_u16>;
using v_int16   = hn::Vec<hwy_d_s16>;
using v_uint32  = hn::Vec<hwy_d_u32>;
using v_int32   = hn::Vec<hwy_d_s32>;
using v_uint64  = hn::Vec<hwy_d_u64>;
using v_int64   = hn::Vec<hwy_d_s64>;
using v_float32 = hn::Vec<hwy_d_f32>;
using v_float64 = hn::Vec<hwy_d_f64>;

// Scalar type aliases expected by existing OpenCV code
using uchar  = unsigned char;
using schar  = signed char;
using ushort = unsigned short;
using uint   = unsigned int;
// Note: int64 and uint64 are NOT redefined here; intrin.hpp already declares
// primary templates v_setall_(int64) and v_setall_(uint64) using cv::int64 and
// cv::uint64 from the OUTER cv namespace. Defining them here would shadow those
// names and break overload resolution for v_setall_ template specialisations.

// ─────────────────────────────────────────────────────────────────────────────
// VTraits specialisations
// ─────────────────────────────────────────────────────────────────────────────
template<class T> struct VTraits;

#define OPENCV_HAL_IMPL_HWY_TRAITS(vec_type, lane_tp, d_type) \
template<> struct VTraits<vec_type> {                          \
    static inline int vlanes() {                               \
        return static_cast<int>(hn::Lanes(d_type()));          \
    }                                                          \
    using lane_type = lane_tp;                                 \
    static const int max_nlanes = HWY_MAX_LANES_D(d_type);    \
};

OPENCV_HAL_IMPL_HWY_TRAITS(v_uint8,   uint8_t,  hwy_d_u8)
OPENCV_HAL_IMPL_HWY_TRAITS(v_int8,    int8_t,   hwy_d_s8)
OPENCV_HAL_IMPL_HWY_TRAITS(v_uint16,  uint16_t, hwy_d_u16)
OPENCV_HAL_IMPL_HWY_TRAITS(v_int16,   int16_t,  hwy_d_s16)
OPENCV_HAL_IMPL_HWY_TRAITS(v_uint32,  uint32_t, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_TRAITS(v_int32,   int32_t,  hwy_d_s32)
OPENCV_HAL_IMPL_HWY_TRAITS(v_uint64,  uint64_t, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_TRAITS(v_int64,   int64_t,  hwy_d_s64)
OPENCV_HAL_IMPL_HWY_TRAITS(v_float32, float,    hwy_d_f32)
OPENCV_HAL_IMPL_HWY_TRAITS(v_float64, double,   hwy_d_f64)

#undef OPENCV_HAL_IMPL_HWY_TRAITS

// ─────────────────────────────────────────────────────────────────────────────
// v_get0  – extract the zeroth (lowest-index) lane as a scalar
// ─────────────────────────────────────────────────────────────────────────────
inline uint8_t  v_get0(const v_uint8&   v) { return hn::ExtractLane(v, 0); }
inline int8_t   v_get0(const v_int8&    v) { return hn::ExtractLane(v, 0); }
inline uint16_t v_get0(const v_uint16&  v) { return hn::ExtractLane(v, 0); }
inline int16_t  v_get0(const v_int16&   v) { return hn::ExtractLane(v, 0); }
inline uint32_t v_get0(const v_uint32&  v) { return hn::ExtractLane(v, 0); }
inline int32_t  v_get0(const v_int32&   v) { return hn::ExtractLane(v, 0); }
inline uint64_t v_get0(const v_uint64&  v) { return hn::ExtractLane(v, 0); }
inline int64_t  v_get0(const v_int64&   v) { return hn::ExtractLane(v, 0); }
inline float    v_get0(const v_float32& v) { return hn::ExtractLane(v, 0); }
inline double   v_get0(const v_float64& v) { return hn::ExtractLane(v, 0); }

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation: v_setzero_*, v_setall_*,  template<> v_setzero_<>, v_setall_<>
// ─────────────────────────────────────────────────────────────────────────────
// Primary templates v_setzero_<> and v_setall_<> are declared in intrin.hpp;
// we only define concrete helpers and provide explicit specialisations here.

#define OPENCV_HAL_IMPL_HWY_INIT(vec_type, lane_tp, suffix, d_type)              \
inline vec_type v_setzero_##suffix() { return hn::Zero(d_type()); }               \
inline vec_type v_setall_##suffix(lane_tp v) { return hn::Set(d_type(), v); }     \
template<> inline vec_type v_setzero_<vec_type>() { return v_setzero_##suffix(); }\
template<> inline vec_type v_setall_<vec_type>(lane_tp v) { return v_setall_##suffix(v); }

OPENCV_HAL_IMPL_HWY_INIT(v_uint8,   uint8_t,  u8,  hwy_d_u8)
OPENCV_HAL_IMPL_HWY_INIT(v_int8,    int8_t,   s8,  hwy_d_s8)
OPENCV_HAL_IMPL_HWY_INIT(v_uint16,  uint16_t, u16, hwy_d_u16)
OPENCV_HAL_IMPL_HWY_INIT(v_int16,   int16_t,  s16, hwy_d_s16)
OPENCV_HAL_IMPL_HWY_INIT(v_uint32,  uint32_t, u32, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_INIT(v_int32,   int32_t,  s32, hwy_d_s32)
// cv::int64 = int64_t = long int on Linux LP64 (via hal/interface.h typedef int64_t int64).
// intrin.hpp declares v_setall_(int64) / v_setall_(uint64) using those typedefs.
// The INIT macro with int64_t/uint64_t as lane_tp generates specialisations that MATCH
// those primary templates, as long as the local int64/uint64 names are NOT shadowed.
OPENCV_HAL_IMPL_HWY_INIT(v_uint64,  uint64_t, u64, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_INIT(v_int64,   int64_t,  s64, hwy_d_s64)
OPENCV_HAL_IMPL_HWY_INIT(v_float32, float,    f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_INIT(v_float64, double,   f64, hwy_d_f64)

#undef OPENCV_HAL_IMPL_HWY_INIT

// ─────────────────────────────────────────────────────────────────────────────
// Reinterpret  (v_reinterpret_as_*)
// Highway's BitCast handles all combinations; the tag carries the target type.
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(vec_type, suffix) \
inline vec_type v_reinterpret_as_##suffix(const vec_type& v) { return v; }

OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_uint8,   u8)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_int8,    s8)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_uint16,  u16)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_int16,   s16)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_uint32,  u32)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_int32,   s32)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_uint64,  u64)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_int64,   s64)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_float32, f32)
OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET(v_float64, f64)

#undef OPENCV_HAL_IMPL_HWY_NOTHING_REINTERPRET

// Cross-type reinterpret: BitCast(dst_tag, src_vec)
#define OPENCV_HAL_IMPL_HWY_REINTERPRET(dst_vec, dst_suffix, dst_d, src_vec, src_suffix) \
inline dst_vec v_reinterpret_as_##dst_suffix(const src_vec& v) {                          \
    return hn::BitCast(dst_d(), v);                                                        \
}                                                                                          \
inline src_vec v_reinterpret_as_##src_suffix(const dst_vec& v) {                          \
    using src_d = hn::RebindToSigned<dst_d>;  /* placeholder – overridden below */        \
    (void)src_d{};                                                                         \
    return hn::BitCast(hn::DFromV<src_vec>(), v);                                         \
}

// Explicit bidirectional pairs
#define OPENCV_HAL_IMPL_HWY_REINT2(v1, s1, d1, v2, s2, d2)       \
inline v1 v_reinterpret_as_##s1(const v2& v) {                    \
    return hn::BitCast(d1(), v);                                   \
}                                                                   \
inline v2 v_reinterpret_as_##s2(const v1& v) {                    \
    return hn::BitCast(d2(), v);                                   \
}

OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_int8,   s8,  hwy_d_s8)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint16, u16, hwy_d_u16, v_int16,  s16, hwy_d_s16)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint32, u32, hwy_d_u32, v_int32,  s32, hwy_d_s32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint64, u64, hwy_d_u64, v_int64,  s64, hwy_d_s64)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint32, u32, hwy_d_u32, v_float32, f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REINT2(v_int32,  s32, hwy_d_s32, v_float32, f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint64, u64, hwy_d_u64, v_float64, f64, hwy_d_f64)
OPENCV_HAL_IMPL_HWY_REINT2(v_int64,  s64, hwy_d_s64, v_float64, f64, hwy_d_f64)

// Same-width-different-signedness (8→16, 8→32, etc.)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_uint16,  u16, hwy_d_u16)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_uint32,  u32, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_uint64,  u64, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint16, u16, hwy_d_u16, v_uint32,  u32, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint16, u16, hwy_d_u16, v_uint64,  u64, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint32, u32, hwy_d_u32, v_uint64,  u64, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_REINT2(v_int8,   s8,  hwy_d_s8,  v_int16,   s16, hwy_d_s16)
OPENCV_HAL_IMPL_HWY_REINT2(v_int8,   s8,  hwy_d_s8,  v_int32,   s32, hwy_d_s32)
OPENCV_HAL_IMPL_HWY_REINT2(v_int8,   s8,  hwy_d_s8,  v_int64,   s64, hwy_d_s64)
OPENCV_HAL_IMPL_HWY_REINT2(v_int16,  s16, hwy_d_s16, v_int32,   s32, hwy_d_s32)
OPENCV_HAL_IMPL_HWY_REINT2(v_int16,  s16, hwy_d_s16, v_int64,   s64, hwy_d_s64)
OPENCV_HAL_IMPL_HWY_REINT2(v_int32,  s32, hwy_d_s32, v_int64,   s64, hwy_d_s64)

// Mixed-sign cross-width pairs (signed ↔ unsigned of different widths)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_int16,  s16, hwy_d_s16)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_int32,  s32, hwy_d_s32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_int64,  s64, hwy_d_s64)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint16, u16, hwy_d_u16, v_int8,   s8,  hwy_d_s8)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint16, u16, hwy_d_u16, v_int32,  s32, hwy_d_s32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint16, u16, hwy_d_u16, v_int64,  s64, hwy_d_s64)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint32, u32, hwy_d_u32, v_int8,   s8,  hwy_d_s8)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint32, u32, hwy_d_u32, v_int16,  s16, hwy_d_s16)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint32, u32, hwy_d_u32, v_int64,  s64, hwy_d_s64)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint64, u64, hwy_d_u64, v_int8,   s8,  hwy_d_s8)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint64, u64, hwy_d_u64, v_int16,  s16, hwy_d_s16)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint64, u64, hwy_d_u64, v_int32,  s32, hwy_d_s32)

// Float ↔ integer cross-width
OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_float32, f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint16, u16, hwy_d_u16, v_float32, f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint64, u64, hwy_d_u64, v_float32, f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REINT2(v_int8,   s8,  hwy_d_s8,  v_float32, f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REINT2(v_int16,  s16, hwy_d_s16, v_float32, f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REINT2(v_int64,  s64, hwy_d_s64, v_float32, f32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint8,  u8,  hwy_d_u8,  v_float64, f64, hwy_d_f64)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint16, u16, hwy_d_u16, v_float64, f64, hwy_d_f64)
OPENCV_HAL_IMPL_HWY_REINT2(v_uint32, u32, hwy_d_u32, v_float64, f64, hwy_d_f64)
OPENCV_HAL_IMPL_HWY_REINT2(v_int8,   s8,  hwy_d_s8,  v_float64, f64, hwy_d_f64)
OPENCV_HAL_IMPL_HWY_REINT2(v_int16,  s16, hwy_d_s16, v_float64, f64, hwy_d_f64)
OPENCV_HAL_IMPL_HWY_REINT2(v_int32,  s32, hwy_d_s32, v_float64, f64, hwy_d_f64)

// float32 ↔ float64  (cross-width floating point)
inline v_float32 v_reinterpret_as_f32(const v_float64& v) {
    return hn::BitCast(hwy_d_f32(), v);
}
inline v_float64 v_reinterpret_as_f64(const v_float32& v) {
    return hn::BitCast(hwy_d_f64(), v);
}

#undef OPENCV_HAL_IMPL_HWY_REINT2

// ─────────────────────────────────────────────────────────────────────────────
// Extract
// ─────────────────────────────────────────────────────────────────────────────
// v_extract<n>(a, b): result = concat(a, b)[n .. n+N-1]
// v_extract_n(v, i):  extract the i-th lane as scalar

#define OPENCV_HAL_IMPL_HWY_EXTRACT(vec_type, lane_tp, d_type)                    \
template<int s = 0>                                                                \
inline vec_type v_extract(const vec_type& a, const vec_type& b, int i = s) {      \
    auto d = d_type();                                                             \
    int N = VTraits<vec_type>::vlanes();                                           \
    /* slide a down by i, then fill with b from position N-i onward */            \
    auto lo = hn::SlideDownLanes(d, a, i);                                        \
    auto hi = hn::SlideUpLanes(d, b, N - i);                                     \
    /* Blend: keep lo in low positions, hi in high positions */                    \
    /* We need the first (N-i) elements from lo and last i elements from hi */    \
    /* Use IfThenElse with an index-based mask */                                  \
    auto idx = hn::Iota(d, 0);                                                    \
    auto threshold = hn::Set(d, static_cast<lane_tp>(N - i));                    \
    auto mask = hn::Lt(idx, threshold);                                           \
    return hn::IfThenElse(mask, lo, hi);                                          \
}                                                                                  \
template<int s = 0>                                                                \
inline lane_tp v_extract_n(const vec_type& v, int i = s) {                       \
    return hn::ExtractLane(v, i);                                                  \
}

// For types where lane_tp is unsigned (needed for Lt comparison with Iota)
#define OPENCV_HAL_IMPL_HWY_EXTRACT_INT(vec_type, lane_tp, d_type, idx_tp, d_idx) \
template<int s = 0>                                                                 \
inline vec_type v_extract(const vec_type& a, const vec_type& b, int i = s) {       \
    auto d = d_type();                                                              \
    int N = VTraits<vec_type>::vlanes();                                            \
    auto lo = hn::SlideDownLanes(d, a, (size_t)i);                                 \
    auto hi = hn::SlideUpLanes(d, b, (size_t)(N - i));                             \
    auto di = d_idx();                                                              \
    auto idx = hn::BitCast(d, hn::Iota(di, 0));                                   \
    auto thresh = hn::Set(d, static_cast<lane_tp>(N - i));                        \
    auto mask = hn::Lt(idx, thresh);                                               \
    return hn::IfThenElse(mask, lo, hi);                                           \
}                                                                                   \
template<int s = 0>                                                                 \
inline lane_tp v_extract_n(const vec_type& v, int i = s) {                        \
    return hn::ExtractLane(v, i);                                                   \
}

// Simpler implementation using SlideDownLanes + SlideUpLanes + blend
#define OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(vec_type, lane_tp, d_type)            \
template<int s = 0>                                                                \
inline vec_type v_extract(const vec_type& a, const vec_type& b, int i = s) {      \
    auto d = d_type();                                                             \
    size_t N = (size_t)VTraits<vec_type>::vlanes();                               \
    if (i == 0) return a;                                                          \
    auto lo = hn::SlideDownLanes(d, a, (size_t)i);                               \
    auto hi = hn::SlideUpLanes(d, b, N - (size_t)i);                             \
    /* Mask: first (N-i) lanes from lo, last i lanes from hi */                   \
    auto mask = hn::FirstN(d, N - (size_t)i);                                    \
    return hn::IfThenElse(mask, lo, hi);                                          \
}                                                                                  \
template<int s = 0>                                                                \
inline lane_tp v_extract_n(const vec_type& v, int i = s) {                       \
    return hn::ExtractLane(v, (size_t)i);                                         \
}

OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_uint8,   uint8_t,  hwy_d_u8)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_int8,    int8_t,   hwy_d_s8)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_uint16,  uint16_t, hwy_d_u16)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_int16,   int16_t,  hwy_d_s16)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_uint32,  uint32_t, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_int32,   int32_t,  hwy_d_s32)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_uint64,  uint64_t, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_int64,   int64_t,  hwy_d_s64)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_float32, float,    hwy_d_f32)
OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE(v_float64, double,   hwy_d_f64)

#undef OPENCV_HAL_IMPL_HWY_EXTRACT_SIMPLE

// v_extract_highest: extract the last lane
#define OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(vec_type, lane_tp)         \
inline lane_tp v_extract_highest(const vec_type& v) {                  \
    return v_extract_n(v, VTraits<vec_type>::vlanes() - 1);            \
}

OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_uint8,   uint8_t)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_int8,    int8_t)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_uint16,  uint16_t)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_int16,   int16_t)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_uint32,  uint32_t)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_int32,   int32_t)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_uint64,  uint64_t)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_int64,   int64_t)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_float32, float)
OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST(v_float64, double)
#undef OPENCV_HAL_IMPL_HWY_EXTRACT_HIGHEST

// ─────────────────────────────────────────────────────────────────────────────
// Load / Store
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_LOADSTORE(vec_type, lane_tp, d_type)                   \
inline vec_type v_load(const lane_tp* ptr) {                                        \
    return hn::LoadU(d_type(), ptr);                                                \
}                                                                                   \
inline vec_type v_load_aligned(const lane_tp* ptr) {                               \
    return hn::Load(d_type(), ptr);                                                 \
}                                                                                   \
inline vec_type v_load_low(const lane_tp* ptr) {                                   \
    /* Load N/2 elements from ptr, zero-extend to full vector */                   \
    auto dh = hn::Half<d_type>();                                                   \
    return hn::ZeroExtendVector(d_type(), hn::LoadU(dh, ptr));                     \
}                                                                                   \
inline vec_type v_load_halves(const lane_tp* ptr0, const lane_tp* ptr1) {          \
    auto dh = hn::Half<d_type>();                                                   \
    auto lo = hn::LoadU(dh, ptr0);                                                  \
    auto hi = hn::LoadU(dh, ptr1);                                                  \
    return hn::Combine(d_type(), hi, lo);                                           \
}                                                                                   \
inline void v_store(lane_tp* ptr, const vec_type& v) {                             \
    hn::StoreU(v, d_type(), ptr);                                                   \
}                                                                                   \
inline void v_store(lane_tp* ptr, const vec_type& v, hal::StoreMode) {             \
    hn::StoreU(v, d_type(), ptr);                                                   \
}                                                                                   \
inline void v_store_aligned(lane_tp* ptr, const vec_type& v) {                     \
    hn::Store(v, d_type(), ptr);                                                    \
}                                                                                   \
inline void v_store_aligned_nocache(lane_tp* ptr, const vec_type& v) {             \
    hn::Store(v, d_type(), ptr);                                                    \
}                                                                                   \
inline void v_store_low(lane_tp* ptr, const vec_type& v) {                        \
    auto dh = hn::Half<d_type>();                                                   \
    hn::StoreU(hn::LowerHalf(dh, v), dh, ptr);                                    \
}                                                                                   \
inline void v_store_high(lane_tp* ptr, const vec_type& v) {                       \
    auto dh = hn::Half<d_type>();                                                   \
    hn::StoreU(hn::UpperHalf(dh, v), dh, ptr);                                    \
}

OPENCV_HAL_IMPL_HWY_LOADSTORE(v_uint8,   uint8_t,  hwy_d_u8)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_int8,    int8_t,   hwy_d_s8)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_uint16,  uint16_t, hwy_d_u16)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_int16,   int16_t,  hwy_d_s16)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_uint32,  uint32_t, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_int32,   int32_t,  hwy_d_s32)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_uint64,  uint64_t, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_int64,   int64_t,  hwy_d_s64)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_float32, float,    hwy_d_f32)
OPENCV_HAL_IMPL_HWY_LOADSTORE(v_float64, double,   hwy_d_f64)

#undef OPENCV_HAL_IMPL_HWY_LOADSTORE

// ─────────────────────────────────────────────────────────────────────────────
// Bridge overloads: on LP64 platforms (AArch64/x86_64 Linux), int64_t = long
// but OpenCV's int64 = long long.  Both are 64-bit but are distinct C++ types,
// so template deduction fails.  Add cast overloads to bridge the gap.
// ─────────────────────────────────────────────────────────────────────────────
#ifdef __LP64__
inline v_int64 v_load(const long long* p)
    { return v_load(reinterpret_cast<const int64_t*>(p)); }
inline v_int64 v_load_aligned(const long long* p)
    { return v_load_aligned(reinterpret_cast<const int64_t*>(p)); }
inline v_int64 v_load_low(const long long* p)
    { return v_load_low(reinterpret_cast<const int64_t*>(p)); }
inline v_int64 v_load_halves(const long long* p0, const long long* p1)
    { return v_load_halves(reinterpret_cast<const int64_t*>(p0),
                           reinterpret_cast<const int64_t*>(p1)); }
inline void v_store(long long* p, const v_int64& v)
    { v_store(reinterpret_cast<int64_t*>(p), v); }
inline void v_store(long long* p, const v_int64& v, hal::StoreMode m)
    { v_store(reinterpret_cast<int64_t*>(p), v, m); }
inline void v_store_aligned(long long* p, const v_int64& v)
    { v_store_aligned(reinterpret_cast<int64_t*>(p), v); }
inline void v_store_aligned_nocache(long long* p, const v_int64& v)
    { v_store_aligned_nocache(reinterpret_cast<int64_t*>(p), v); }
inline void v_store_low(long long* p, const v_int64& v)
    { v_store_low(reinterpret_cast<int64_t*>(p), v); }
inline void v_store_high(long long* p, const v_int64& v)
    { v_store_high(reinterpret_cast<int64_t*>(p), v); }

inline v_uint64 v_load(const unsigned long long* p)
    { return v_load(reinterpret_cast<const uint64_t*>(p)); }
inline v_uint64 v_load_aligned(const unsigned long long* p)
    { return v_load_aligned(reinterpret_cast<const uint64_t*>(p)); }
inline v_uint64 v_load_low(const unsigned long long* p)
    { return v_load_low(reinterpret_cast<const uint64_t*>(p)); }
inline v_uint64 v_load_halves(const unsigned long long* p0, const unsigned long long* p1)
    { return v_load_halves(reinterpret_cast<const uint64_t*>(p0),
                           reinterpret_cast<const uint64_t*>(p1)); }
inline void v_store(unsigned long long* p, const v_uint64& v)
    { v_store(reinterpret_cast<uint64_t*>(p), v); }
inline void v_store(unsigned long long* p, const v_uint64& v, hal::StoreMode m)
    { v_store(reinterpret_cast<uint64_t*>(p), v, m); }
inline void v_store_aligned(unsigned long long* p, const v_uint64& v)
    { v_store_aligned(reinterpret_cast<uint64_t*>(p), v); }
inline void v_store_aligned_nocache(unsigned long long* p, const v_uint64& v)
    { v_store_aligned_nocache(reinterpret_cast<uint64_t*>(p), v); }
inline void v_store_low(unsigned long long* p, const v_uint64& v)
    { v_store_low(reinterpret_cast<uint64_t*>(p), v); }
inline void v_store_high(unsigned long long* p, const v_uint64& v)
    { v_store_high(reinterpret_cast<uint64_t*>(p), v); }
#endif  // __LP64__

// ─────────────────────────────────────────────────────────────────────────────
// Lookup table (LUT) – gather from a scalar table using integer indices
// ─────────────────────────────────────────────────────────────────────────────
inline v_int8 v_lut(const schar* tab, const int* idx) {
    // GatherIndex requires index element size == data element size (1 byte for int8).
    // Since we have 32-bit int indices, use a scalar loop instead.
    auto d8 = hwy_d_s8();
    int N = VTraits<v_int8>::vlanes();
    schar buf[HWY_MAX_LANES_D(hwy_d_s8)];
    for (int i = 0; i < N; i++) buf[i] = tab[idx[i]];
    return hn::LoadU(d8, buf);
}

inline v_int16 v_lut(const short* tab, const int* idx) {
    // GatherIndex requires index element size == data element size (2 bytes for int16).
    // Since we have 32-bit int indices, use a scalar loop instead.
    auto d16 = hwy_d_s16();
    int N = VTraits<v_int16>::vlanes();
    short buf[HWY_MAX_LANES_D(hwy_d_s16)];
    for (int i = 0; i < N; i++) buf[i] = tab[idx[i]];
    return hn::LoadU(d16, buf);
}

inline v_int32 v_lut(const int* tab, const int* idx) {
    auto d32 = hwy_d_s32();
    auto vidx = hn::LoadU(d32, idx);
    return hn::GatherIndex(d32, tab, vidx);
}

inline v_int64 v_lut(const int64_t* tab, const int* idx) {
    // GatherIndex requires 64-bit indices for 64-bit data; use scalar loop.
    auto d64 = hwy_d_s64();
    int N = VTraits<v_int64>::vlanes();
    int64_t buf[HWY_MAX_LANES_D(hwy_d_s64)];
    for (int i = 0; i < N; i++) buf[i] = tab[idx[i]];
    return hn::LoadU(d64, buf);
}

inline v_float32 v_lut(const float* tab, const int* idx) {
    auto d32 = hwy_d_f32();
    auto di32 = hwy_d_s32();
    auto vidx = hn::LoadU(di32, idx);
    return hn::GatherIndex(d32, tab, vidx);
}

inline v_float64 v_lut(const double* tab, const int* idx) {
    // GatherIndex for f64 requires 64-bit indices; use scalar loop.
    auto d64 = hwy_d_f64();
    int N = VTraits<v_float64>::vlanes();
    double buf[HWY_MAX_LANES_D(hwy_d_f64)];
    for (int i = 0; i < N; i++) buf[i] = tab[idx[i]];
    return hn::LoadU(d64, buf);
}

// Unsigned wrappers delegating to signed
inline v_uint8  v_lut(const uchar* tab, const int* idx) {
    return v_reinterpret_as_u8(v_lut((const schar*)tab, idx));
}
inline v_uint16 v_lut(const ushort* tab, const int* idx) {
    return v_reinterpret_as_u16(v_lut((const short*)tab, idx));
}
inline v_uint32 v_lut(const unsigned* tab, const int* idx) {
    return v_reinterpret_as_u32(v_lut((const int*)tab, idx));
}
inline v_uint64 v_lut(const uint64* tab, const int* idx) {
    return v_reinterpret_as_u64(v_lut((const int64_t*)tab, idx));
}

// v_lut with vector indices
inline v_float32 v_lut(const float* tab, const v_int32& vidx) {
    auto d = hwy_d_f32();
    return hn::GatherIndex(d, tab, vidx);
}
inline v_int32 v_lut(const int* tab, const v_int32& vidx) {
    auto d = hwy_d_s32();
    return hn::GatherIndex(d, tab, vidx);
}
inline v_uint32 v_lut(const unsigned* tab, const v_int32& vidx) {
    return v_reinterpret_as_u32(v_lut((const int*)tab, vidx));
}
inline v_float64 v_lut(const double* tab, const v_int32& vidx) {
    // v_float64 has N/2 lanes vs v_int32's N lanes. Use lower N/2 int32 indices,
    // promote them to int64 (same lane count as v_float64), then GatherIndex.
    auto df    = hwy_d_f64();            // full f64 descriptor (N/2 lanes on NEON: 2)
    auto di64  = hwy_d_s64();            // full i64 descriptor (same N/2 lanes)
    auto dh32  = hn::Half<hwy_d_s32>(); // half of i32: N/2 lanes = 2 on NEON
    auto idx32 = hn::LowerHalf(dh32, vidx);  // take lower N/2 int32 indices
    auto idx64 = hn::PromoteTo(di64, idx32); // promote N/2 int32 → N/2 int64
    return hn::GatherIndex(df, tab, idx64);
}

// v_lut_pairs: load pairs from table, indices are {idx[i], idx[i]+1}
inline v_int8 v_lut_pairs(const schar* tab, const int* idx) {
    auto d8 = hwy_d_s8();
    int N = VTraits<v_int8>::vlanes();
    // Scalar fallback for simplicity
    schar buf[HWY_MAX_LANES_D(hwy_d_s8)];
    for (int i = 0; i < N / 2; i++) {
        buf[2*i]     = tab[idx[i]];
        buf[2*i + 1] = tab[idx[i] + 1];
    }
    return hn::LoadU(d8, buf);
}
inline v_int16 v_lut_pairs(const short* tab, const int* idx) {
    auto d16 = hwy_d_s16();
    int N = VTraits<v_int16>::vlanes();
    short buf[HWY_MAX_LANES_D(hwy_d_s16)];
    for (int i = 0; i < N / 2; i++) {
        buf[2*i]     = tab[idx[i]];
        buf[2*i + 1] = tab[idx[i] + 1];
    }
    return hn::LoadU(d16, buf);
}
inline v_int32 v_lut_pairs(const int* tab, const int* idx) {
    auto d32 = hwy_d_s32();
    int N = VTraits<v_int32>::vlanes();
    int buf[HWY_MAX_LANES_D(hwy_d_s32)];
    for (int i = 0; i < N / 2; i++) {
        buf[2*i]     = tab[idx[i]];
        buf[2*i + 1] = tab[idx[i] + 1];
    }
    return hn::LoadU(d32, buf);
}
inline v_float32 v_lut_pairs(const float* tab, const int* idx) {
    auto d32 = hwy_d_f32();
    int N = VTraits<v_float32>::vlanes();
    float buf[HWY_MAX_LANES_D(hwy_d_f32)];
    for (int i = 0; i < N / 2; i++) {
        buf[2*i]     = tab[idx[i]];
        buf[2*i + 1] = tab[idx[i] + 1];
    }
    return hn::LoadU(d32, buf);
}
inline v_int64 v_lut_pairs(const int64_t* tab, const int* idx) {
    auto d64 = hwy_d_s64();
    int N = VTraits<v_int64>::vlanes();
    int64_t buf[HWY_MAX_LANES_D(hwy_d_s64)];
    for (int i = 0; i < N / 2; i++) {
        buf[2*i]     = tab[idx[i]];
        buf[2*i + 1] = tab[idx[i] + 1];
    }
    return hn::LoadU(d64, buf);
}
inline v_float64 v_lut_pairs(const double* tab, const int* idx) {
    auto d64 = hwy_d_f64();
    int N = VTraits<v_float64>::vlanes();
    double buf[HWY_MAX_LANES_D(hwy_d_f64)];
    for (int i = 0; i < N / 2; i++) {
        buf[2*i]     = tab[idx[i]];
        buf[2*i + 1] = tab[idx[i] + 1];
    }
    return hn::LoadU(d64, buf);
}
inline v_uint8  v_lut_pairs(const uchar* tab, const int* idx) {
    return v_reinterpret_as_u8(v_lut_pairs((const schar*)tab, idx));
}
inline v_uint16 v_lut_pairs(const ushort* tab, const int* idx) {
    return v_reinterpret_as_u16(v_lut_pairs((const short*)tab, idx));
}
inline v_uint32 v_lut_pairs(const unsigned* tab, const int* idx) {
    return v_reinterpret_as_u32(v_lut_pairs((const int*)tab, idx));
}
inline v_uint64 v_lut_pairs(const uint64* tab, const int* idx) {
    return v_reinterpret_as_u64(v_lut_pairs((const int64_t*)tab, idx));
}

// v_lut_quads: load groups of 4 from table
inline v_int8 v_lut_quads(const schar* tab, const int* idx) {
    auto d8 = hwy_d_s8();
    int N = VTraits<v_int8>::vlanes();
    schar buf[HWY_MAX_LANES_D(hwy_d_s8)];
    for (int i = 0; i < N / 4; i++)
        for (int j = 0; j < 4; j++)
            buf[4*i + j] = tab[idx[i] + j];
    return hn::LoadU(d8, buf);
}
inline v_int16 v_lut_quads(const short* tab, const int* idx) {
    auto d16 = hwy_d_s16();
    int N = VTraits<v_int16>::vlanes();
    short buf[HWY_MAX_LANES_D(hwy_d_s16)];
    for (int i = 0; i < N / 4; i++)
        for (int j = 0; j < 4; j++)
            buf[4*i + j] = tab[idx[i] + j];
    return hn::LoadU(d16, buf);
}
inline v_int32 v_lut_quads(const int* tab, const int* idx) {
    auto d32 = hwy_d_s32();
    int N = VTraits<v_int32>::vlanes();
    int buf[HWY_MAX_LANES_D(hwy_d_s32)];
    for (int i = 0; i < N / 4; i++)
        for (int j = 0; j < 4; j++)
            buf[4*i + j] = tab[idx[i] + j];
    return hn::LoadU(d32, buf);
}
inline v_float32 v_lut_quads(const float* tab, const int* idx) {
    auto d32 = hwy_d_f32();
    int N = VTraits<v_float32>::vlanes();
    float buf[HWY_MAX_LANES_D(hwy_d_f32)] = {0};
    for (int i = 0; i < N / 4; i++)
        for (int j = 0; j < 4; j++)
            buf[4*i + j] = tab[idx[i] + j];
    return hn::LoadU(d32, buf);
}
inline v_uint8  v_lut_quads(const uchar* tab, const int* idx) {
    return v_reinterpret_as_u8(v_lut_quads((const schar*)tab, idx));
}
inline v_uint16 v_lut_quads(const ushort* tab, const int* idx) {
    return v_reinterpret_as_u16(v_lut_quads((const short*)tab, idx));
}
inline v_uint32 v_lut_quads(const unsigned* tab, const int* idx) {
    return v_reinterpret_as_u32(v_lut_quads((const int*)tab, idx));
}

// v_lut_deinterleave: gather (x,y) pairs from table
inline void v_lut_deinterleave(const float* tab, const v_int32& vidx, v_float32& vx, v_float32& vy) {
    auto d = hwy_d_f32();
    auto di = hwy_d_s32();
    // offsets for x: vidx * 2, for y: vidx * 2 + 1
    auto scale = hn::Set(di, 2);
    auto vidx2 = hn::Mul(vidx, scale);
    auto vidx2p1 = hn::Add(vidx2, hn::Set(di, 1));
    vx = hn::GatherIndex(d, tab, vidx2);
    vy = hn::GatherIndex(d, tab, vidx2p1);
}
inline void v_lut_deinterleave(const int* tab, const v_int32& vidx, v_int32& vx, v_int32& vy) {
    auto d = hwy_d_s32();
    auto scale = hn::Set(d, 2);
    auto vidx2 = hn::Mul(vidx, scale);
    auto vidx2p1 = hn::Add(vidx2, hn::Set(d, 1));
    vx = hn::GatherIndex(d, tab, vidx2);
    vy = hn::GatherIndex(d, tab, vidx2p1);
}
inline void v_lut_deinterleave(const unsigned* tab, const v_int32& vidx, v_uint32& vx, v_uint32& vy) {
    v_int32 ix, iy;
    v_lut_deinterleave((const int*)tab, vidx, ix, iy);
    vx = v_reinterpret_as_u32(ix);
    vy = v_reinterpret_as_u32(iy);
}
inline void v_lut_deinterleave(const double* tab, const v_int32& vidx,
                                v_float64& vx, v_float64& vy) {
    // v_float64 has N/2 lanes. Use bottom N/2 int32 indices, promote to int64.
    auto df      = hwy_d_f64();            // full f64 descriptor (N/2 lanes)
    auto di64    = hwy_d_s64();            // full i64 descriptor (same N/2 lanes)
    auto dh32    = hn::Half<hwy_d_s32>(); // N/2 i32 lanes (lower half of vidx)
    auto scale64 = hn::Set(di64, int64_t(2));
    auto idx32   = hn::LowerHalf(dh32, vidx);
    auto idx64   = hn::PromoteTo(di64, idx32);
    auto idx64_x2   = hn::Mul(idx64, scale64);
    auto idx64_x2p1 = hn::Add(idx64_x2, hn::Set(di64, int64_t(1)));
    vx = hn::GatherIndex(df, tab, idx64_x2);
    vy = hn::GatherIndex(df, tab, idx64_x2p1);
}

// LP64 bridge overloads: v_lut / v_lut_pairs for OpenCV int64 (= long long).
// On LP64 Linux, int64_t = long int but OpenCV int64 = long long; bridges needed.
// Note: uint64 (= unsigned long long) already has wrappers via the uint64 alias.
// Note: v_lut_quads is not defined for int64/uint64, so no bridge needed there.
#ifdef __LP64__
inline v_int64 v_lut(const long long* tab, const int* idx)
    { return v_lut(reinterpret_cast<const int64_t*>(tab), idx); }
inline v_int64 v_lut_pairs(const long long* tab, const int* idx)
    { return v_lut_pairs(reinterpret_cast<const int64_t*>(tab), idx); }
#endif  // __LP64__ (v_lut int64 bridges)

// ─────────────────────────────────────────────────────────────────────────────
// Pack boolean
// ─────────────────────────────────────────────────────────────────────────────
inline v_uint8 v_pack_b(const v_uint16& a, const v_uint16& b) {
    auto d8 = hwy_d_u8();
    // Narrow u16 to u8 (keep low byte)
    return hn::OrderedDemote2To(d8, a, b);
}

inline v_uint8 v_pack_b(const v_uint32& a, const v_uint32& b,
                         const v_uint32& c, const v_uint32& d) {
    auto d8  = hwy_d_u8();
    auto d16 = hwy_d_u16();
    auto ab = hn::OrderedDemote2To(d16, a, b);
    auto cd = hn::OrderedDemote2To(d16, c, d);
    return hn::OrderedDemote2To(d8, ab, cd);
}

inline v_uint8 v_pack_b(const v_uint64& a, const v_uint64& b, const v_uint64& c,
                         const v_uint64& d, const v_uint64& e, const v_uint64& f,
                         const v_uint64& g, const v_uint64& h) {
    auto d8  = hwy_d_u8();
    auto d16 = hwy_d_u16();
    auto d32 = hwy_d_u32();
    auto ab = hn::OrderedDemote2To(d32, a, b);
    auto cd = hn::OrderedDemote2To(d32, c, d);
    auto ef = hn::OrderedDemote2To(d32, e, f);
    auto gh = hn::OrderedDemote2To(d32, g, h);
    auto abcd = hn::OrderedDemote2To(d16, ab, cd);
    auto efgh = hn::OrderedDemote2To(d16, ef, gh);
    return hn::OrderedDemote2To(d8, abcd, efgh);
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic  (saturating for 8/16-bit, wrapping for 32/64-bit, IEEE for fp)
// ─────────────────────────────────────────────────────────────────────────────

// ── Saturating add / sub (u8, i8, u16, i16) ────────────────────────────────
inline v_uint8  v_add(const v_uint8&  a, const v_uint8&  b) { return hn::SaturatedAdd(a, b); }
inline v_uint8  v_sub(const v_uint8&  a, const v_uint8&  b) { return hn::SaturatedSub(a, b); }
inline v_int8   v_add(const v_int8&   a, const v_int8&   b) { return hn::SaturatedAdd(a, b); }
inline v_int8   v_sub(const v_int8&   a, const v_int8&   b) { return hn::SaturatedSub(a, b); }
inline v_uint16 v_add(const v_uint16& a, const v_uint16& b) { return hn::SaturatedAdd(a, b); }
inline v_uint16 v_sub(const v_uint16& a, const v_uint16& b) { return hn::SaturatedSub(a, b); }
inline v_int16  v_add(const v_int16&  a, const v_int16&  b) { return hn::SaturatedAdd(a, b); }
inline v_int16  v_sub(const v_int16&  a, const v_int16&  b) { return hn::SaturatedSub(a, b); }

// ── Wrapping add / sub / mul  (u32, i32, u64, i64) ─────────────────────────
inline v_uint32 v_add(const v_uint32& a, const v_uint32& b) { return hn::Add(a, b); }
inline v_uint32 v_sub(const v_uint32& a, const v_uint32& b) { return hn::Sub(a, b); }
inline v_uint32 v_mul(const v_uint32& a, const v_uint32& b) { return hn::Mul(a, b); }
inline v_int32  v_add(const v_int32&  a, const v_int32&  b) { return hn::Add(a, b); }
inline v_int32  v_sub(const v_int32&  a, const v_int32&  b) { return hn::Sub(a, b); }
inline v_int32  v_mul(const v_int32&  a, const v_int32&  b) { return hn::Mul(a, b); }
inline v_uint64 v_add(const v_uint64& a, const v_uint64& b) { return hn::Add(a, b); }
inline v_uint64 v_sub(const v_uint64& a, const v_uint64& b) { return hn::Sub(a, b); }
inline v_int64  v_add(const v_int64&  a, const v_int64&  b) { return hn::Add(a, b); }
inline v_int64  v_sub(const v_int64&  a, const v_int64&  b) { return hn::Sub(a, b); }

// ── Float arithmetic ────────────────────────────────────────────────────────
inline v_float32 v_add(const v_float32& a, const v_float32& b) { return hn::Add(a, b); }
inline v_float32 v_sub(const v_float32& a, const v_float32& b) { return hn::Sub(a, b); }
inline v_float32 v_mul(const v_float32& a, const v_float32& b) { return hn::Mul(a, b); }
inline v_float32 v_div(const v_float32& a, const v_float32& b) { return hn::Div(a, b); }
inline v_float64 v_add(const v_float64& a, const v_float64& b) { return hn::Add(a, b); }
inline v_float64 v_sub(const v_float64& a, const v_float64& b) { return hn::Sub(a, b); }
inline v_float64 v_mul(const v_float64& a, const v_float64& b) { return hn::Mul(a, b); }
inline v_float64 v_div(const v_float64& a, const v_float64& b) { return hn::Div(a, b); }

// ── Saturating multiply  (u8, i8, u16, i16): widen → mul → clamp → narrow ──
// Helper: u8 saturating multiply
inline v_uint8 v_mul(const v_uint8& a, const v_uint8& b) {
    auto d8  = hwy_d_u8();
    auto d16 = hn::RepartitionToWide<hwy_d_u8>();
    auto a_lo = hn::PromoteLowerTo(d16, a);
    auto b_lo = hn::PromoteLowerTo(d16, b);
    auto a_hi = hn::PromoteUpperTo(d16, a);
    auto b_hi = hn::PromoteUpperTo(d16, b);
    auto prod_lo = hn::Mul(a_lo, b_lo);
    auto prod_hi = hn::Mul(a_hi, b_hi);
    auto max_u8  = hn::Set(d16, uint16_t(255));
    prod_lo = hn::Min(prod_lo, max_u8);
    prod_hi = hn::Min(prod_hi, max_u8);
    return hn::OrderedDemote2To(d8, prod_lo, prod_hi);
}
inline v_int8 v_mul(const v_int8& a, const v_int8& b) {
    auto d8  = hwy_d_s8();
    auto d16 = hn::RepartitionToWide<hwy_d_s8>();
    auto a_lo = hn::PromoteLowerTo(d16, a);
    auto b_lo = hn::PromoteLowerTo(d16, b);
    auto a_hi = hn::PromoteUpperTo(d16, a);
    auto b_hi = hn::PromoteUpperTo(d16, b);
    auto prod_lo = hn::Mul(a_lo, b_lo);
    auto prod_hi = hn::Mul(a_hi, b_hi);
    auto min_i8  = hn::Set(d16, int16_t(-128));
    auto max_i8  = hn::Set(d16, int16_t(127));
    prod_lo = hn::Clamp(prod_lo, min_i8, max_i8);
    prod_hi = hn::Clamp(prod_hi, min_i8, max_i8);
    return hn::OrderedDemote2To(d8, prod_lo, prod_hi);
}
inline v_uint16 v_mul(const v_uint16& a, const v_uint16& b) {
    auto d16 = hwy_d_u16();
    auto d32 = hn::RepartitionToWide<hwy_d_u16>();
    auto a_lo = hn::PromoteLowerTo(d32, a);
    auto b_lo = hn::PromoteLowerTo(d32, b);
    auto a_hi = hn::PromoteUpperTo(d32, a);
    auto b_hi = hn::PromoteUpperTo(d32, b);
    auto prod_lo = hn::Mul(a_lo, b_lo);
    auto prod_hi = hn::Mul(a_hi, b_hi);
    auto max_u16 = hn::Set(d32, uint32_t(65535));
    prod_lo = hn::Min(prod_lo, max_u16);
    prod_hi = hn::Min(prod_hi, max_u16);
    return hn::OrderedDemote2To(d16, prod_lo, prod_hi);
}
inline v_int16 v_mul(const v_int16& a, const v_int16& b) {
    auto d16 = hwy_d_s16();
    auto d32 = hn::RepartitionToWide<hwy_d_s16>();
    auto a_lo = hn::PromoteLowerTo(d32, a);
    auto b_lo = hn::PromoteLowerTo(d32, b);
    auto a_hi = hn::PromoteUpperTo(d32, a);
    auto b_hi = hn::PromoteUpperTo(d32, b);
    auto prod_lo = hn::Mul(a_lo, b_lo);
    auto prod_hi = hn::Mul(a_hi, b_hi);
    auto min_i16 = hn::Set(d32, int32_t(-32768));
    auto max_i16 = hn::Set(d32, int32_t(32767));
    prod_lo = hn::Clamp(prod_lo, min_i16, max_i16);
    prod_hi = hn::Clamp(prod_hi, min_i16, max_i16);
    return hn::OrderedDemote2To(d16, prod_lo, prod_hi);
}

// ── Variadic v_add / v_mul  (3+ arguments) ─────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_BIN_MADD(vec_type, add_expr)                          \
template<typename... Args>                                                          \
inline vec_type v_add(const vec_type& a, const vec_type& b, const Args&... rest) { \
    return v_add((add_expr), rest...);                                             \
}

OPENCV_HAL_IMPL_HWY_BIN_MADD(v_uint8,   hn::SaturatedAdd(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_int8,    hn::SaturatedAdd(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_uint16,  hn::SaturatedAdd(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_int16,   hn::SaturatedAdd(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_uint32,  hn::Add(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_int32,   hn::Add(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_uint64,  hn::Add(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_int64,   hn::Add(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_float32, hn::Add(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MADD(v_float64, hn::Add(a, b))
#undef OPENCV_HAL_IMPL_HWY_BIN_MADD

#define OPENCV_HAL_IMPL_HWY_BIN_MMUL(vec_type, mul_expr)                          \
template<typename... Args>                                                          \
inline vec_type v_mul(const vec_type& a, const vec_type& b, const Args&... rest) { \
    return v_mul((mul_expr), rest...);                                             \
}
OPENCV_HAL_IMPL_HWY_BIN_MMUL(v_uint8,   v_mul(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MMUL(v_int8,    v_mul(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MMUL(v_uint16,  v_mul(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MMUL(v_int16,   v_mul(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MMUL(v_uint32,  hn::Mul(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MMUL(v_int32,   hn::Mul(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MMUL(v_float32, hn::Mul(a, b))
OPENCV_HAL_IMPL_HWY_BIN_MMUL(v_float64, hn::Mul(a, b))
#undef OPENCV_HAL_IMPL_HWY_BIN_MMUL

// ── mul_expand: widening multiply ───────────────────────────────────────────
inline void v_mul_expand(const v_uint8& a, const v_uint8& b, v_uint16& c, v_uint16& d) {
    auto d16 = hn::RepartitionToWide<hwy_d_u8>();
    c = hn::Mul(hn::PromoteLowerTo(d16, a), hn::PromoteLowerTo(d16, b));
    d = hn::Mul(hn::PromoteUpperTo(d16, a), hn::PromoteUpperTo(d16, b));
}
inline void v_mul_expand(const v_int8& a, const v_int8& b, v_int16& c, v_int16& d) {
    auto d16 = hn::RepartitionToWide<hwy_d_s8>();
    c = hn::Mul(hn::PromoteLowerTo(d16, a), hn::PromoteLowerTo(d16, b));
    d = hn::Mul(hn::PromoteUpperTo(d16, a), hn::PromoteUpperTo(d16, b));
}
inline void v_mul_expand(const v_uint16& a, const v_uint16& b, v_uint32& c, v_uint32& d) {
    auto d32 = hn::RepartitionToWide<hwy_d_u16>();
    c = hn::Mul(hn::PromoteLowerTo(d32, a), hn::PromoteLowerTo(d32, b));
    d = hn::Mul(hn::PromoteUpperTo(d32, a), hn::PromoteUpperTo(d32, b));
}
inline void v_mul_expand(const v_int16& a, const v_int16& b, v_int32& c, v_int32& d) {
    auto d32 = hn::RepartitionToWide<hwy_d_s16>();
    c = hn::Mul(hn::PromoteLowerTo(d32, a), hn::PromoteLowerTo(d32, b));
    d = hn::Mul(hn::PromoteUpperTo(d32, a), hn::PromoteUpperTo(d32, b));
}
inline void v_mul_expand(const v_uint32& a, const v_uint32& b, v_uint64& c, v_uint64& d) {
    // MulEven/MulOdd interleave by even/odd indices; recombine to lower/upper half order
    // even[k] = a[2k]*b[2k], odd[k] = a[2k+1]*b[2k+1]
    const hwy_d_u64 d64;
    auto even = hn::MulEven(a, b);
    auto odd  = hn::MulOdd(a, b);
#if HWY_HAVE_SCALABLE
    // Scalable targets (RVV/SVE): vectors may span multiple 128-bit blocks.
    // Use half-split approach to get correct ordering across blocks.
    // When VLEN==128 (d64h has only 1 lane), fall back to full-vector path.
    const hn::Half<hwy_d_u64> d64h;
    if (hn::Lanes(d64h) <= 1) {
        c = hn::InterleaveLower(d64, even, odd);
        d = hn::InterleaveUpper(d64, even, odd);
    } else {
        auto even_lo = hn::LowerHalf(d64h, even);
        auto odd_lo  = hn::LowerHalf(d64h, odd);
        auto even_hi = hn::UpperHalf(d64h, even);
        auto odd_hi  = hn::UpperHalf(d64h, odd);
        c = hn::Combine(d64, hn::InterleaveUpper(d64h, even_lo, odd_lo),
                             hn::InterleaveLower(d64h, even_lo, odd_lo));
        d = hn::Combine(d64, hn::InterleaveUpper(d64h, even_hi, odd_hi),
                             hn::InterleaveLower(d64h, even_hi, odd_hi));
    }
#else
    // Fixed-width targets (NEON/SSE etc.): always exactly 128-bit (one block).
    // Half<hwy_d_u64> = Vec128<uint64_t, 1> which lacks InterleaveLower/Upper on NEON.
    // The full-vector InterleaveLower/Upper is correct for a single 128-bit block.
    c = hn::InterleaveLower(d64, even, odd);
    d = hn::InterleaveUpper(d64, even, odd);
#endif
}

// ── mul_hi: return high 16 bits of 16-bit multiply ──────────────────────────
inline v_int16  v_mul_hi(const v_int16&  a, const v_int16&  b) { return hn::MulHigh(a, b); }
inline v_uint16 v_mul_hi(const v_uint16& a, const v_uint16& b) { return hn::MulHigh(a, b); }

// ── Wrapping arithmetic (add_wrap, sub_wrap, mul_wrap) ───────────────────────
inline v_uint8  v_add_wrap(const v_uint8&  a, const v_uint8&  b) { return hn::Add(a, b); }
inline v_uint8  v_sub_wrap(const v_uint8&  a, const v_uint8&  b) { return hn::Sub(a, b); }
inline v_int8   v_add_wrap(const v_int8&   a, const v_int8&   b) { return hn::Add(a, b); }
inline v_int8   v_sub_wrap(const v_int8&   a, const v_int8&   b) { return hn::Sub(a, b); }
inline v_uint16 v_add_wrap(const v_uint16& a, const v_uint16& b) { return hn::Add(a, b); }
inline v_uint16 v_sub_wrap(const v_uint16& a, const v_uint16& b) { return hn::Sub(a, b); }
inline v_int16  v_add_wrap(const v_int16&  a, const v_int16&  b) { return hn::Add(a, b); }
inline v_int16  v_sub_wrap(const v_int16&  a, const v_int16&  b) { return hn::Sub(a, b); }

// 8/16-bit wrapping multiply: widen, multiply, mask low bits
inline v_uint8 v_mul_wrap(const v_uint8& a, const v_uint8& b) {
    auto d8  = hwy_d_u8();
    auto d16 = hn::RepartitionToWide<hwy_d_u8>();
    auto mask = hn::Set(d16, uint16_t(0xFF));
    auto lo = hn::And(hn::Mul(hn::PromoteLowerTo(d16, a), hn::PromoteLowerTo(d16, b)), mask);
    auto hi = hn::And(hn::Mul(hn::PromoteUpperTo(d16, a), hn::PromoteUpperTo(d16, b)), mask);
    return hn::OrderedDemote2To(d8, lo, hi);
}
inline v_int8 v_mul_wrap(const v_int8& a, const v_int8& b) {
    // Use unsigned path to avoid signed int16→int8 saturation in OrderedDemote2To.
    // BitCast to u8, widen to u16, multiply, AND 0xFF (truncate), then demote via u8 path.
    const hwy_d_s8 d8s;
    const hwy_d_u8 d8u;
    const auto d16u = hn::RepartitionToWide<hwy_d_u8>();
    auto au = hn::BitCast(d8u, a);
    auto bu = hn::BitCast(d8u, b);
    auto mask = hn::Set(d16u, uint16_t(0xFF));
    auto lo = hn::And(hn::Mul(hn::PromoteLowerTo(d16u, au),
                               hn::PromoteLowerTo(d16u, bu)), mask);
    auto hi = hn::And(hn::Mul(hn::PromoteUpperTo(d16u, au),
                               hn::PromoteUpperTo(d16u, bu)), mask);
    return hn::BitCast(d8s, hn::OrderedDemote2To(d8u, lo, hi));
}
inline v_uint16 v_mul_wrap(const v_uint16& a, const v_uint16& b) { return hn::Mul(a, b); }
inline v_int16  v_mul_wrap(const v_int16&  a, const v_int16&  b) { return hn::Mul(a, b); }

// ─────────────────────────────────────────────────────────────────────────────
// Bitwise logic
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_LOGIC_INT(vec_type)                     \
inline vec_type v_and(const vec_type& a, const vec_type& b) {       \
    return hn::And(a, b);                                            \
}                                                                    \
inline vec_type v_or(const vec_type& a, const vec_type& b) {        \
    return hn::Or(a, b);                                             \
}                                                                    \
inline vec_type v_xor(const vec_type& a, const vec_type& b) {       \
    return hn::Xor(a, b);                                            \
}                                                                    \
inline vec_type v_not(const vec_type& a) {                           \
    return hn::Not(a);                                               \
}

OPENCV_HAL_IMPL_HWY_LOGIC_INT(v_uint8)
OPENCV_HAL_IMPL_HWY_LOGIC_INT(v_int8)
OPENCV_HAL_IMPL_HWY_LOGIC_INT(v_uint16)
OPENCV_HAL_IMPL_HWY_LOGIC_INT(v_int16)
OPENCV_HAL_IMPL_HWY_LOGIC_INT(v_uint32)
OPENCV_HAL_IMPL_HWY_LOGIC_INT(v_int32)
OPENCV_HAL_IMPL_HWY_LOGIC_INT(v_uint64)
OPENCV_HAL_IMPL_HWY_LOGIC_INT(v_int64)
#undef OPENCV_HAL_IMPL_HWY_LOGIC_INT

// Float bitwise via reinterpret
inline v_float32 v_and(const v_float32& a, const v_float32& b) {
    return v_reinterpret_as_f32(v_and(v_reinterpret_as_s32(a), v_reinterpret_as_s32(b)));
}
inline v_float32 v_or(const v_float32& a, const v_float32& b) {
    return v_reinterpret_as_f32(v_or(v_reinterpret_as_s32(a), v_reinterpret_as_s32(b)));
}
inline v_float32 v_xor(const v_float32& a, const v_float32& b) {
    return v_reinterpret_as_f32(v_xor(v_reinterpret_as_s32(a), v_reinterpret_as_s32(b)));
}
inline v_float32 v_not(const v_float32& a) {
    return v_reinterpret_as_f32(v_not(v_reinterpret_as_s32(a)));
}
inline v_float64 v_and(const v_float64& a, const v_float64& b) {
    return v_reinterpret_as_f64(v_and(v_reinterpret_as_s64(a), v_reinterpret_as_s64(b)));
}
inline v_float64 v_or(const v_float64& a, const v_float64& b) {
    return v_reinterpret_as_f64(v_or(v_reinterpret_as_s64(a), v_reinterpret_as_s64(b)));
}
inline v_float64 v_xor(const v_float64& a, const v_float64& b) {
    return v_reinterpret_as_f64(v_xor(v_reinterpret_as_s64(a), v_reinterpret_as_s64(b)));
}
inline v_float64 v_not(const v_float64& a) {
    return v_reinterpret_as_f64(v_not(v_reinterpret_as_s64(a)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit shifts
// ─────────────────────────────────────────────────────────────────────────────
// Unsigned: logical shift right
#define OPENCV_HAL_IMPL_HWY_UNSIGNED_SHIFT(vec_type)                         \
template<int s = 0>                                                           \
inline vec_type v_shl(const vec_type& a, int n = s) {                        \
    return hn::ShiftLeftSame(a, n);                                           \
}                                                                             \
template<int s = 0>                                                           \
inline vec_type v_shr(const vec_type& a, int n = s) {                        \
    return hn::ShiftRightSame(a, n);                                          \
}

// Signed: arithmetic shift right
#define OPENCV_HAL_IMPL_HWY_SIGNED_SHIFT(vec_type)                           \
template<int s = 0>                                                           \
inline vec_type v_shl(const vec_type& a, int n = s) {                        \
    return hn::ShiftLeftSame(a, n);                                           \
}                                                                             \
template<int s = 0>                                                           \
inline vec_type v_shr(const vec_type& a, int n = s) {                        \
    return hn::ShiftRightSame(a, n);                                          \
}

OPENCV_HAL_IMPL_HWY_UNSIGNED_SHIFT(v_uint16)
OPENCV_HAL_IMPL_HWY_UNSIGNED_SHIFT(v_uint32)
OPENCV_HAL_IMPL_HWY_UNSIGNED_SHIFT(v_uint64)
OPENCV_HAL_IMPL_HWY_SIGNED_SHIFT(v_int16)
OPENCV_HAL_IMPL_HWY_SIGNED_SHIFT(v_int32)
OPENCV_HAL_IMPL_HWY_SIGNED_SHIFT(v_int64)

#undef OPENCV_HAL_IMPL_HWY_UNSIGNED_SHIFT
#undef OPENCV_HAL_IMPL_HWY_SIGNED_SHIFT

// ─────────────────────────────────────────────────────────────────────────────
// Comparison  (return all-1s / all-0s vector per lane)
// Highway comparisons return Mask; we convert via VecFromMask.
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_CMP(vec_type, d_type)                                \
inline vec_type v_eq(const vec_type& a, const vec_type& b) {                     \
    return hn::VecFromMask(d_type(), hn::Eq(a, b));                              \
}                                                                                  \
inline vec_type v_ne(const vec_type& a, const vec_type& b) {                     \
    return hn::VecFromMask(d_type(), hn::Ne(a, b));                              \
}                                                                                  \
inline vec_type v_lt(const vec_type& a, const vec_type& b) {                     \
    return hn::VecFromMask(d_type(), hn::Lt(a, b));                              \
}                                                                                  \
inline vec_type v_gt(const vec_type& a, const vec_type& b) {                     \
    return hn::VecFromMask(d_type(), hn::Gt(a, b));                              \
}                                                                                  \
inline vec_type v_le(const vec_type& a, const vec_type& b) {                     \
    return hn::VecFromMask(d_type(), hn::Le(a, b));                              \
}                                                                                  \
inline vec_type v_ge(const vec_type& a, const vec_type& b) {                     \
    return hn::VecFromMask(d_type(), hn::Ge(a, b));                              \
}

OPENCV_HAL_IMPL_HWY_CMP(v_uint8,   hwy_d_u8)
OPENCV_HAL_IMPL_HWY_CMP(v_int8,    hwy_d_s8)
OPENCV_HAL_IMPL_HWY_CMP(v_uint16,  hwy_d_u16)
OPENCV_HAL_IMPL_HWY_CMP(v_int16,   hwy_d_s16)
OPENCV_HAL_IMPL_HWY_CMP(v_uint32,  hwy_d_u32)
OPENCV_HAL_IMPL_HWY_CMP(v_int32,   hwy_d_s32)
OPENCV_HAL_IMPL_HWY_CMP(v_uint64,  hwy_d_u64)
OPENCV_HAL_IMPL_HWY_CMP(v_int64,   hwy_d_s64)
OPENCV_HAL_IMPL_HWY_CMP(v_float32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_CMP(v_float64, hwy_d_f64)
#undef OPENCV_HAL_IMPL_HWY_CMP

inline v_float32 v_not_nan(const v_float32& a) { return v_eq(a, a); }
inline v_float64 v_not_nan(const v_float64& a) { return v_eq(a, a); }

// ─────────────────────────────────────────────────────────────────────────────
// Min / Max
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_MINMAX(vec_type)                              \
inline vec_type v_min(const vec_type& a, const vec_type& b) {            \
    return hn::Min(a, b);                                                 \
}                                                                          \
inline vec_type v_max(const vec_type& a, const vec_type& b) {            \
    return hn::Max(a, b);                                                 \
}

OPENCV_HAL_IMPL_HWY_MINMAX(v_uint8)
OPENCV_HAL_IMPL_HWY_MINMAX(v_int8)
OPENCV_HAL_IMPL_HWY_MINMAX(v_uint16)
OPENCV_HAL_IMPL_HWY_MINMAX(v_int16)
OPENCV_HAL_IMPL_HWY_MINMAX(v_uint32)
OPENCV_HAL_IMPL_HWY_MINMAX(v_int32)
OPENCV_HAL_IMPL_HWY_MINMAX(v_float32)
OPENCV_HAL_IMPL_HWY_MINMAX(v_float64)
#undef OPENCV_HAL_IMPL_HWY_MINMAX

// ─────────────────────────────────────────────────────────────────────────────
// Absolute value / absdiff / absdiffs
// ─────────────────────────────────────────────────────────────────────────────
// absdiff for unsigned: max(a,b) - min(a,b)
inline v_uint8  v_absdiff(const v_uint8&  a, const v_uint8&  b) {
    return v_sub(v_max(a, b), v_min(a, b));
}
inline v_uint16 v_absdiff(const v_uint16& a, const v_uint16& b) {
    return v_sub(v_max(a, b), v_min(a, b));
}
inline v_uint32 v_absdiff(const v_uint32& a, const v_uint32& b) {
    return v_sub(v_max(a, b), v_min(a, b));
}
inline v_float32 v_absdiff(const v_float32& a, const v_float32& b) {
    return hn::Abs(hn::Sub(a, b));
}
inline v_float64 v_absdiff(const v_float64& a, const v_float64& b) {
    return hn::Abs(hn::Sub(a, b));
}

// absdiff for signed ints: result is UNSIGNED, clamped to positive
inline v_uint8 v_absdiff(const v_int8& a, const v_int8& b) {
    // Use wider type, sub, abs, narrow
    auto d16 = hn::RepartitionToWide<hwy_d_s8>();
    auto du8 = hwy_d_u8();
    auto a_lo = hn::PromoteLowerTo(d16, a);
    auto b_lo = hn::PromoteLowerTo(d16, b);
    auto a_hi = hn::PromoteUpperTo(d16, a);
    auto b_hi = hn::PromoteUpperTo(d16, b);
    auto diff_lo = hn::Abs(hn::Sub(a_lo, b_lo));
    auto diff_hi = hn::Abs(hn::Sub(a_hi, b_hi));
    // Narrow: values are [0,128], safe to truncate
    return hn::OrderedDemote2To(du8, hn::BitCast(hn::RepartitionToWide<hwy_d_u8>(), diff_lo),
                                     hn::BitCast(hn::RepartitionToWide<hwy_d_u8>(), diff_hi));
}
inline v_uint16 v_absdiff(const v_int16& a, const v_int16& b) {
    auto d32 = hn::RepartitionToWide<hwy_d_s16>();
    auto du16 = hwy_d_u16();
    auto a_lo = hn::PromoteLowerTo(d32, a);
    auto b_lo = hn::PromoteLowerTo(d32, b);
    auto a_hi = hn::PromoteUpperTo(d32, a);
    auto b_hi = hn::PromoteUpperTo(d32, b);
    auto diff_lo = hn::Abs(hn::Sub(a_lo, b_lo));
    auto diff_hi = hn::Abs(hn::Sub(a_hi, b_hi));
    return hn::OrderedDemote2To(du16, hn::BitCast(hn::RepartitionToWide<hwy_d_u16>(), diff_lo),
                                       hn::BitCast(hn::RepartitionToWide<hwy_d_u16>(), diff_hi));
}
inline v_uint32 v_absdiff(const v_int32& a, const v_int32& b) {
    auto d64 = hn::RepartitionToWide<hwy_d_s32>();
    auto du32 = hwy_d_u32();
    auto a_lo = hn::PromoteLowerTo(d64, a);
    auto b_lo = hn::PromoteLowerTo(d64, b);
    auto a_hi = hn::PromoteUpperTo(d64, a);
    auto b_hi = hn::PromoteUpperTo(d64, b);
    auto diff_lo = hn::Abs(hn::Sub(a_lo, b_lo));
    auto diff_hi = hn::Abs(hn::Sub(a_hi, b_hi));
    return hn::OrderedDemote2To(du32, hn::BitCast(hn::RepartitionToWide<hwy_d_u32>(), diff_lo),
                                       hn::BitCast(hn::RepartitionToWide<hwy_d_u32>(), diff_hi));
}

// absdiffs: saturating absolute difference (result stays signed)
inline v_int8  v_absdiffs(const v_int8&  a, const v_int8&  b) {
    return v_sub(v_max(a, b), v_min(a, b));
}
inline v_int16 v_absdiffs(const v_int16& a, const v_int16& b) {
    return v_sub(v_max(a, b), v_min(a, b));
}

// abs: signed integer absolute value (result is unsigned)
inline v_uint8  v_abs(const v_int8&    a) { return v_absdiff(a, v_setzero_s8()); }
inline v_uint16 v_abs(const v_int16&   a) { return v_absdiff(a, v_setzero_s16()); }
inline v_uint32 v_abs(const v_int32&   a) { return v_absdiff(a, v_setzero_s32()); }
inline v_float32 v_abs(const v_float32& a) { return hn::Abs(a); }
inline v_float64 v_abs(const v_float64& a) { return hn::Abs(a); }

// ─────────────────────────────────────────────────────────────────────────────
// Reduce
// ─────────────────────────────────────────────────────────────────────────────
inline unsigned v_reduce_sum(const v_uint8&  a) {
    // hn::ReduceSum in uint8 may overflow (max sum = N*255);
    // promote to uint16 first to avoid overflow
    const hwy_d_u16 d16;
    auto lo16 = hn::PromoteLowerTo(d16, a);  // lower N/2 lanes widened to uint16
    auto hi16 = hn::PromoteUpperTo(d16, a);  // upper N/2 lanes widened to uint16
    return (unsigned)hn::ReduceSum(d16, hn::Add(lo16, hi16));
}
inline int      v_reduce_sum(const v_int8&   a) {
    // hn::ReduceSum in int8 may overflow (max sum = N*127);
    // promote to int16 first to avoid overflow
    auto d16 = hwy_d_s16();
    auto lo  = hn::PromoteLowerTo(d16, a);   // lower N/2 lanes widened to int16
    auto hi  = hn::PromoteUpperTo(d16, a);   // upper N/2 lanes widened to int16
    return (int)hn::ReduceSum(d16, hn::Add(lo, hi));
}
inline unsigned v_reduce_sum(const v_uint16& a) {
    return (unsigned)hn::ReduceSum(hwy_d_u16(), a);
}
inline int      v_reduce_sum(const v_int16&  a) {
    return (int)hn::ReduceSum(hwy_d_s16(), a);
}
inline unsigned v_reduce_sum(const v_uint32& a) {
    return (unsigned)hn::ReduceSum(hwy_d_u32(), a);
}
inline int      v_reduce_sum(const v_int32&  a) {
    return (int)hn::ReduceSum(hwy_d_s32(), a);
}
inline uint64   v_reduce_sum(const v_uint64& a) {
    return (uint64)hn::ReduceSum(hwy_d_u64(), a);
}
inline int64    v_reduce_sum(const v_int64&  a) {
    return (int64)hn::ReduceSum(hwy_d_s64(), a);
}
inline float    v_reduce_sum(const v_float32& a) {
    return (float)hn::ReduceSum(hwy_d_f32(), a);
}
inline double   v_reduce_sum(const v_float64& a) {
    return (double)hn::ReduceSum(hwy_d_f64(), a);
}

inline uchar  v_reduce_min(const v_uint8&   a) { return (uchar) hn::ReduceMin(hwy_d_u8(),  a); }
inline schar  v_reduce_min(const v_int8&    a) { return (schar) hn::ReduceMin(hwy_d_s8(),  a); }
inline ushort v_reduce_min(const v_uint16&  a) { return (ushort)hn::ReduceMin(hwy_d_u16(), a); }
inline short  v_reduce_min(const v_int16&   a) { return (short) hn::ReduceMin(hwy_d_s16(), a); }
inline uint   v_reduce_min(const v_uint32&  a) { return (uint)  hn::ReduceMin(hwy_d_u32(), a); }
inline int    v_reduce_min(const v_int32&   a)  { return (int)  hn::ReduceMin(hwy_d_s32(), a); }
inline float  v_reduce_min(const v_float32& a)  { return (float)hn::ReduceMin(hwy_d_f32(), a); }
inline double v_reduce_min(const v_float64& a)  { return (double)hn::ReduceMin(hwy_d_f64(), a); }

inline uchar  v_reduce_max(const v_uint8&   a) { return (uchar) hn::ReduceMax(hwy_d_u8(),  a); }
inline schar  v_reduce_max(const v_int8&    a) { return (schar) hn::ReduceMax(hwy_d_s8(),  a); }
inline ushort v_reduce_max(const v_uint16&  a) { return (ushort)hn::ReduceMax(hwy_d_u16(), a); }
inline short  v_reduce_max(const v_int16&   a) { return (short) hn::ReduceMax(hwy_d_s16(), a); }
inline uint   v_reduce_max(const v_uint32&  a) { return (uint)  hn::ReduceMax(hwy_d_u32(), a); }
inline int    v_reduce_max(const v_int32&   a) { return (int)   hn::ReduceMax(hwy_d_s32(), a); }
inline float  v_reduce_max(const v_float32& a) { return (float) hn::ReduceMax(hwy_d_f32(), a); }
inline double v_reduce_max(const v_float64& a) { return (double)hn::ReduceMax(hwy_d_f64(), a); }

// ── reduce_sad ───────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_REDUCE_SAD(vec_type, scalar_type)          \
inline scalar_type v_reduce_sad(const vec_type& a, const vec_type& b) {\
    return v_reduce_sum(v_absdiff(a, b));                              \
}
OPENCV_HAL_IMPL_HWY_REDUCE_SAD(v_uint8,   unsigned)
OPENCV_HAL_IMPL_HWY_REDUCE_SAD(v_int8,    unsigned)
OPENCV_HAL_IMPL_HWY_REDUCE_SAD(v_uint16,  unsigned)
OPENCV_HAL_IMPL_HWY_REDUCE_SAD(v_int16,   unsigned)
OPENCV_HAL_IMPL_HWY_REDUCE_SAD(v_uint32,  unsigned)
OPENCV_HAL_IMPL_HWY_REDUCE_SAD(v_int32,   unsigned)
OPENCV_HAL_IMPL_HWY_REDUCE_SAD(v_float32, float)
#undef OPENCV_HAL_IMPL_HWY_REDUCE_SAD

// ── v_reduce_sum4 ────────────────────────────────────────────────────────────
inline v_float32 v_reduce_sum4(const v_float32& a, const v_float32& b,
                                const v_float32& c, const v_float32& d) {
    // When VLEN>128, we have more than 4 lanes; compute partial sums per group of 4.
    // Each group of 4 lanes in the result holds the sum of the corresponding 4 lanes
    // from a, b, c, d respectively.
    const hwy_d_f32 df;
    const int n = (int)hn::Lanes(df);
    float buf_a[HWY_MAX_LANES_D(hwy_d_f32)];
    float buf_b[HWY_MAX_LANES_D(hwy_d_f32)];
    float buf_c[HWY_MAX_LANES_D(hwy_d_f32)];
    float buf_d[HWY_MAX_LANES_D(hwy_d_f32)];
    float res[HWY_MAX_LANES_D(hwy_d_f32)] = {};
    hn::Store(a, df, buf_a);
    hn::Store(b, df, buf_b);
    hn::Store(c, df, buf_c);
    hn::Store(d, df, buf_d);
    for (int i = 0; i < n; i += 4) {
        res[i]     = buf_a[i] + buf_a[i+1] + buf_a[i+2] + buf_a[i+3];
        res[i + 1] = buf_b[i] + buf_b[i+1] + buf_b[i+2] + buf_b[i+3];
        res[i + 2] = buf_c[i] + buf_c[i+1] + buf_c[i+2] + buf_c[i+3];
        res[i + 3] = buf_d[i] + buf_d[i+1] + buf_d[i+2] + buf_d[i+3];
    }
    return hn::Load(df, res);
}

// ─────────────────────────────────────────────────────────────────────────────
// Zip4 / Transpose4x4  (work only when N >= 4, which is true for ≥128-bit SIMD)
// ─────────────────────────────────────────────────────────────────────────────
inline void v_zip4(const v_uint32& a0, const v_uint32& a1, v_uint32& b0, v_uint32& b1) {
    auto d = hwy_d_u32();
    b0 = hn::InterleaveLower(d, a0, a1);
    b1 = hn::InterleaveUpper(d, a0, a1);
}
inline void v_zip4(const v_int32& a0, const v_int32& a1, v_int32& b0, v_int32& b1) {
    auto d = hwy_d_s32();
    b0 = hn::InterleaveLower(d, a0, a1);
    b1 = hn::InterleaveUpper(d, a0, a1);
}
inline void v_zip4(const v_float32& a0, const v_float32& a1, v_float32& b0, v_float32& b1) {
    auto d = hwy_d_f32();
    b0 = hn::InterleaveLower(d, a0, a1);
    b1 = hn::InterleaveUpper(d, a0, a1);
}

#define OPENCV_HAL_IMPL_HWY_TRANSPOSE4x4(vec_type)                            \
inline void v_transpose4x4(const vec_type& a0, const vec_type& a1,            \
                            const vec_type& a2, const vec_type& a3,            \
                            vec_type& b0, vec_type& b1,                        \
                            vec_type& b2, vec_type& b3) {                      \
    vec_type t0, t1, t2, t3;                                                   \
    v_zip4(a0, a2, t0, t2);                                                    \
    v_zip4(a1, a3, t1, t3);                                                    \
    v_zip4(t0, t1, b0, b1);                                                    \
    v_zip4(t2, t3, b2, b3);                                                    \
}
OPENCV_HAL_IMPL_HWY_TRANSPOSE4x4(v_uint32)
OPENCV_HAL_IMPL_HWY_TRANSPOSE4x4(v_int32)
OPENCV_HAL_IMPL_HWY_TRANSPOSE4x4(v_float32)
#undef OPENCV_HAL_IMPL_HWY_TRANSPOSE4x4

// ─────────────────────────────────────────────────────────────────────────────
// Square-root, invsqrt, magnitude, sqr_magnitude
// ─────────────────────────────────────────────────────────────────────────────
inline v_float32 v_sqrt(const v_float32& x) { return hn::Sqrt(x); }
inline v_float64 v_sqrt(const v_float64& x) { return hn::Sqrt(x); }

inline v_float32 v_invsqrt(const v_float32& x) {
    return v_div(v_setall_f32(1.0f), v_sqrt(x));
}
inline v_float64 v_invsqrt(const v_float64& x) {
    return v_div(v_setall_f64(1.0), v_sqrt(x));
}

inline v_float32 v_magnitude(const v_float32& a, const v_float32& b) {
    return v_sqrt(hn::MulAdd(a, a, hn::Mul(b, b)));
}
inline v_float32 v_sqr_magnitude(const v_float32& a, const v_float32& b) {
    return hn::MulAdd(a, a, hn::Mul(b, b));
}
inline v_float64 v_magnitude(const v_float64& a, const v_float64& b) {
    return v_sqrt(hn::MulAdd(a, a, hn::Mul(b, b)));
}
inline v_float64 v_sqr_magnitude(const v_float64& a, const v_float64& b) {
    return hn::MulAdd(a, a, hn::Mul(b, b));
}

// ─────────────────────────────────────────────────────────────────────────────
// FMA / muladd
// ─────────────────────────────────────────────────────────────────────────────
inline v_float32 v_fma(const v_float32& a, const v_float32& b, const v_float32& c) {
    return hn::MulAdd(a, b, c);
}
inline v_float64 v_fma(const v_float64& a, const v_float64& b, const v_float64& c) {
    return hn::MulAdd(a, b, c);
}
inline v_int32 v_fma(const v_int32& a, const v_int32& b, const v_int32& c) {
    return hn::Add(hn::Mul(a, b), c);
}

inline v_float32 v_muladd(const v_float32& a, const v_float32& b, const v_float32& c) {
    return v_fma(a, b, c);
}
inline v_float64 v_muladd(const v_float64& a, const v_float64& b, const v_float64& c) {
    return v_fma(a, b, c);
}
inline v_int32 v_muladd(const v_int32& a, const v_int32& b, const v_int32& c) {
    return v_fma(a, b, c);
}

// ─────────────────────────────────────────────────────────────────────────────
// Check all / any  (check sign bits)
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_CHECK_ALLANY(vec_type, d_type)             \
inline bool v_check_all(const vec_type& a) {                            \
    auto d = d_type();                                                  \
    auto zero = hn::Zero(d);                                            \
    return hn::AllTrue(d, hn::Lt(a, zero));                            \
}                                                                       \
inline bool v_check_any(const vec_type& a) {                            \
    auto d = d_type();                                                  \
    auto zero = hn::Zero(d);                                            \
    return !hn::AllFalse(d, hn::Lt(a, zero));                         \
}

OPENCV_HAL_IMPL_HWY_CHECK_ALLANY(v_int8,  hwy_d_s8)
OPENCV_HAL_IMPL_HWY_CHECK_ALLANY(v_int16, hwy_d_s16)
OPENCV_HAL_IMPL_HWY_CHECK_ALLANY(v_int32, hwy_d_s32)
OPENCV_HAL_IMPL_HWY_CHECK_ALLANY(v_int64, hwy_d_s64)
#undef OPENCV_HAL_IMPL_HWY_CHECK_ALLANY

inline bool v_check_all(const v_uint8&   a) { return v_check_all(v_reinterpret_as_s8(a)); }
inline bool v_check_any(const v_uint8&   a) { return v_check_any(v_reinterpret_as_s8(a)); }
inline bool v_check_all(const v_uint16&  a) { return v_check_all(v_reinterpret_as_s16(a)); }
inline bool v_check_any(const v_uint16&  a) { return v_check_any(v_reinterpret_as_s16(a)); }
inline bool v_check_all(const v_uint32&  a) { return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_uint32&  a) { return v_check_any(v_reinterpret_as_s32(a)); }
inline bool v_check_all(const v_uint64&  a) { return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_uint64&  a) { return v_check_any(v_reinterpret_as_s64(a)); }
inline bool v_check_all(const v_float32& a) { return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_float32& a) { return v_check_any(v_reinterpret_as_s32(a)); }
inline bool v_check_all(const v_float64& a) { return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_float64& a) { return v_check_any(v_reinterpret_as_s64(a)); }

// ─────────────────────────────────────────────────────────────────────────────
// Select  (mask ? a : b  where mask uses sign-bit convention)
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_SELECT(vec_type, d_type)                           \
inline vec_type v_select(const vec_type& mask, const vec_type& a,              \
                          const vec_type& b) {                                  \
    auto d = d_type();                                                          \
    auto zero = hn::Zero(d);                                                   \
    /* mask lanes are -1 (true) or 0 (false); treat as "not-equal-to-zero" */ \
    return hn::IfThenElse(hn::Ne(mask, zero), a, b);                          \
}

OPENCV_HAL_IMPL_HWY_SELECT(v_uint8,   hwy_d_u8)
OPENCV_HAL_IMPL_HWY_SELECT(v_uint16,  hwy_d_u16)
OPENCV_HAL_IMPL_HWY_SELECT(v_uint32,  hwy_d_u32)
OPENCV_HAL_IMPL_HWY_SELECT(v_int8,    hwy_d_s8)
OPENCV_HAL_IMPL_HWY_SELECT(v_int16,   hwy_d_s16)
OPENCV_HAL_IMPL_HWY_SELECT(v_int32,   hwy_d_s32)
#undef OPENCV_HAL_IMPL_HWY_SELECT

inline v_float32 v_select(const v_float32& mask, const v_float32& a,
                           const v_float32& b) {
    auto d = hwy_d_f32();
    auto zero = hn::Zero(d);
    return hn::IfThenElse(hn::Ne(mask, zero), a, b);
}
inline v_float64 v_select(const v_float64& mask, const v_float64& a,
                           const v_float64& b) {
    auto d = hwy_d_f64();
    auto zero = hn::Zero(d);
    return hn::IfThenElse(hn::Ne(mask, zero), a, b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotate shift  (v_rotate_right / v_rotate_left)
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_ROTATE(vec_type, lane_tp, d_type)                   \
template<int n> inline vec_type v_rotate_right(const vec_type& a) {             \
    auto d = d_type();                                                           \
    return hn::SlideDownLanes(d, a, (size_t)n);                                 \
}                                                                                \
template<int n> inline vec_type v_rotate_left(const vec_type& a) {              \
    auto d = d_type();                                                           \
    return hn::SlideUpLanes(d, a, (size_t)n);                                  \
}                                                                                \
template<> inline vec_type v_rotate_left<0>(const vec_type& a) { return a; }   \
template<int n> inline vec_type v_rotate_right(const vec_type& a,               \
                                                const vec_type& b) {             \
    /* result = upper (N-n) lanes of a, then lower n lanes of b */              \
    auto d = d_type();                                                           \
    size_t N = (size_t)VTraits<vec_type>::vlanes();                             \
    auto lo = hn::SlideDownLanes(d, a, (size_t)n);                             \
    auto hi = hn::SlideUpLanes(d, b, N - (size_t)n);                          \
    auto mask = hn::FirstN(d, N - (size_t)n);                                 \
    return hn::IfThenElse(mask, lo, hi);                                        \
}                                                                                \
template<int n> inline vec_type v_rotate_left(const vec_type& a,                \
                                               const vec_type& b) {              \
    auto d = d_type();                                                           \
    size_t N = (size_t)VTraits<vec_type>::vlanes();                             \
    auto lo = hn::SlideDownLanes(d, b, N - (size_t)n);                        \
    auto hi = hn::SlideUpLanes(d, a, (size_t)n);                              \
    auto mask = hn::FirstN(d, (size_t)n);                                      \
    return hn::IfThenElse(mask, lo, hi);                                        \
}                                                                                \
template<> inline vec_type v_rotate_left<0>(const vec_type& a,                  \
                                             const vec_type& b) {                \
    (void)b; return a;                                                           \
}

OPENCV_HAL_IMPL_HWY_ROTATE(v_uint8,   uint8_t,  hwy_d_u8)
OPENCV_HAL_IMPL_HWY_ROTATE(v_int8,    int8_t,   hwy_d_s8)
OPENCV_HAL_IMPL_HWY_ROTATE(v_uint16,  uint16_t, hwy_d_u16)
OPENCV_HAL_IMPL_HWY_ROTATE(v_int16,   int16_t,  hwy_d_s16)
OPENCV_HAL_IMPL_HWY_ROTATE(v_uint32,  uint32_t, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_ROTATE(v_int32,   int32_t,  hwy_d_s32)
OPENCV_HAL_IMPL_HWY_ROTATE(v_uint64,  uint64_t, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_ROTATE(v_int64,   int64_t,  hwy_d_s64)
OPENCV_HAL_IMPL_HWY_ROTATE(v_float32, float,    hwy_d_f32)
OPENCV_HAL_IMPL_HWY_ROTATE(v_float64, double,   hwy_d_f64)
#undef OPENCV_HAL_IMPL_HWY_ROTATE

// ─────────────────────────────────────────────────────────────────────────────
// Type conversion: float ↔ int
// ─────────────────────────────────────────────────────────────────────────────
inline v_float32 v_cvt_f32(const v_int32& a) {
    return hn::ConvertTo(hwy_d_f32(), a);
}

inline v_float32 v_cvt_f32(const v_float64& a) {
    // Take lower N/2 f64 lanes, demote to f32 (N/2 lanes)
    auto dh32 = hn::Half<hwy_d_f32>();
    return hn::ZeroExtendVector(hwy_d_f32(), hn::DemoteTo(dh32, a));
}

inline v_float32 v_cvt_f32(const v_float64& a, const v_float64& b) {
    // a → lower N/2 f32 lanes, b → upper N/2 f32 lanes
    auto df = hwy_d_f32();
    auto dh32 = hn::Half<hwy_d_f32>();
    auto lo = hn::DemoteTo(dh32, a);
    auto hi = hn::DemoteTo(dh32, b);
    return hn::Combine(df, hi, lo);
}

inline v_float64 v_cvt_f64(const v_int32& a) {
    // Promote lower N/2 i32 lanes to i64, then convert to f64
    auto d64 = hwy_d_f64();
    auto d_i64 = hwy_d_s64();
    auto dh32 = hn::Half<hwy_d_s32>();
    auto half = hn::LowerHalf(dh32, a);
    return hn::ConvertTo(d64, hn::PromoteTo(d_i64, half));
}

inline v_float64 v_cvt_f64_high(const v_int32& a) {
    auto d64 = hwy_d_f64();
    auto d_i64 = hwy_d_s64();
    auto dh32 = hn::Half<hwy_d_s32>();
    auto half = hn::UpperHalf(dh32, a);
    return hn::ConvertTo(d64, hn::PromoteTo(d_i64, half));
}

inline v_float64 v_cvt_f64(const v_float32& a) {
    auto d64 = hwy_d_f64();
    auto dh32 = hn::Half<hwy_d_f32>();
    auto half = hn::LowerHalf(dh32, a);
    return hn::PromoteTo(d64, half);
}

inline v_float64 v_cvt_f64_high(const v_float32& a) {
    auto d64 = hwy_d_f64();
    auto dh32 = hn::Half<hwy_d_f32>();
    auto half = hn::UpperHalf(dh32, a);
    return hn::PromoteTo(d64, half);
}

inline v_float64 v_cvt_f64(const v_int64& a) {
    return hn::ConvertTo(hwy_d_f64(), a);
}

// ─────────────────────────────────────────────────────────────────────────────
// Broadcast element
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_BROADCAST(vec_type, lane_tp, suffix, d_type)       \
template<int s = 0>                                                              \
inline vec_type v_broadcast_element(const vec_type& v, int i = s) {            \
    return v_setall_##suffix(v_extract_n(v, i));                               \
}                                                                                \
inline vec_type v_broadcast_highest(const vec_type& v) {                       \
    return v_setall_##suffix(v_extract_n(v, VTraits<vec_type>::vlanes() - 1)); \
}

OPENCV_HAL_IMPL_HWY_BROADCAST(v_uint32,  uint32_t, u32, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_BROADCAST(v_int32,   int32_t,  s32, hwy_d_s32)
OPENCV_HAL_IMPL_HWY_BROADCAST(v_float32, float,    f32, hwy_d_f32)
#undef OPENCV_HAL_IMPL_HWY_BROADCAST

// ─────────────────────────────────────────────────────────────────────────────
// Reverse
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_REVERSE(vec_type, d_type)               \
inline vec_type v_reverse(const vec_type& a) {                       \
    return hn::Reverse(d_type(), a);                                 \
}

OPENCV_HAL_IMPL_HWY_REVERSE(v_uint8,   hwy_d_u8)
OPENCV_HAL_IMPL_HWY_REVERSE(v_int8,    hwy_d_s8)
OPENCV_HAL_IMPL_HWY_REVERSE(v_uint16,  hwy_d_u16)
OPENCV_HAL_IMPL_HWY_REVERSE(v_int16,   hwy_d_s16)
OPENCV_HAL_IMPL_HWY_REVERSE(v_uint32,  hwy_d_u32)
OPENCV_HAL_IMPL_HWY_REVERSE(v_int32,   hwy_d_s32)
OPENCV_HAL_IMPL_HWY_REVERSE(v_uint64,  hwy_d_u64)
OPENCV_HAL_IMPL_HWY_REVERSE(v_int64,   hwy_d_s64)
OPENCV_HAL_IMPL_HWY_REVERSE(v_float32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_REVERSE(v_float64, hwy_d_f64)
#undef OPENCV_HAL_IMPL_HWY_REVERSE

// ─────────────────────────────────────────────────────────────────────────────
// Expand: narrow → wide  (v_expand, v_expand_low, v_expand_high, v_load_expand)
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_EXPAND(narrow_vec, wide_vec, narrow_d, wide_d, narrow_tp, wide_tp) \
inline void v_expand(const narrow_vec& a, wide_vec& b0, wide_vec& b1) {                        \
    auto dw = wide_d();                                                                         \
    b0 = hn::PromoteLowerTo(dw, a);                                                            \
    b1 = hn::PromoteUpperTo(dw, a);                                                            \
}                                                                                               \
inline wide_vec v_expand_low(const narrow_vec& a) {                                            \
    return hn::PromoteLowerTo(wide_d(), a);                                                    \
}                                                                                               \
inline wide_vec v_expand_high(const narrow_vec& a) {                                           \
    return hn::PromoteUpperTo(wide_d(), a);                                                    \
}                                                                                               \
inline wide_vec v_load_expand(const narrow_tp* ptr) {                                          \
    /* Load N/2 narrow elements → N/2 wide elements */                                         \
    auto dw = wide_d();                                                                         \
    auto dhn = hn::Half<narrow_d>();                                                            \
    return hn::PromoteTo(dw, hn::LoadU(dhn, ptr));                                             \
}

OPENCV_HAL_IMPL_HWY_EXPAND(v_uint8,  v_uint16, hwy_d_u8,  hwy_d_u16, uint8_t,  uint16_t)
OPENCV_HAL_IMPL_HWY_EXPAND(v_int8,   v_int16,  hwy_d_s8,  hwy_d_s16, int8_t,   int16_t)
OPENCV_HAL_IMPL_HWY_EXPAND(v_uint16, v_uint32, hwy_d_u16, hwy_d_u32, uint16_t, uint32_t)
OPENCV_HAL_IMPL_HWY_EXPAND(v_int16,  v_int32,  hwy_d_s16, hwy_d_s32, int16_t,  int32_t)
OPENCV_HAL_IMPL_HWY_EXPAND(v_uint32, v_uint64, hwy_d_u32, hwy_d_u64, uint32_t, uint64_t)
OPENCV_HAL_IMPL_HWY_EXPAND(v_int32,  v_int64,  hwy_d_s32, hwy_d_s64, int32_t,  int64_t)
#undef OPENCV_HAL_IMPL_HWY_EXPAND

// v_load_expand_q: load 4x-narrower and expand by 4
inline v_uint32 v_load_expand_q(const uchar* ptr) {
    auto du32 = hwy_d_u32();
    int N = VTraits<v_uint32>::vlanes();
    uint8_t buf[HWY_MAX_LANES_D(hwy_d_u32)];
    memcpy(buf, ptr, (size_t)N);
    uint32_t out[HWY_MAX_LANES_D(hwy_d_u32)];
    for (int i = 0; i < N; i++) out[i] = (uint32_t)buf[i];
    return hn::LoadU(du32, out);
}
inline v_int32 v_load_expand_q(const schar* ptr) {
    auto di32 = hwy_d_s32();
    int N = VTraits<v_int32>::vlanes();
    int32_t out[HWY_MAX_LANES_D(hwy_d_s32)] = {0};
    for (int i = 0; i < N; i++) out[i] = (int32_t)ptr[i];
    return hn::LoadU(di32, out);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pack (narrow with saturation): v_pack, v_pack_store, v_rshr_pack, etc.
// ─────────────────────────────────────────────────────────────────────────────
// u16 → u8 (saturating)
inline v_uint8 v_pack(const v_uint16& a, const v_uint16& b) {
    auto d8  = hwy_d_u8();
    auto d16 = hwy_d_u16();
    auto max8 = hn::Set(d16, uint16_t(255));
    return hn::OrderedDemote2To(d8, hn::Min(a, max8), hn::Min(b, max8));
}
inline void v_pack_store(uchar* ptr, const v_uint16& a) {
    auto dh8 = hn::Half<hwy_d_u8>();
    auto d16 = hwy_d_u16();
    auto max8 = hn::Set(d16, uint16_t(255));
    hn::StoreU(hn::DemoteTo(dh8, hn::Min(a, max8)), dh8, ptr);
}
template<int n = 0>
inline v_uint8 v_rshr_pack(const v_uint16& a, const v_uint16& b, int N = n) {
    auto d8  = hwy_d_u8();
    auto d16 = hwy_d_u16();
    // Round-shift right by N
    auto half = hn::Set(d16, uint16_t(1 << std::max(0, N-1)));
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto bs = hn::ShiftRightSame(hn::Add(b, half), N);
    auto max8 = hn::Set(d16, uint16_t(255));
    return hn::OrderedDemote2To(d8, hn::Min(as, max8), hn::Min(bs, max8));
}
template<int n = 0>
inline void v_rshr_pack_store(uchar* ptr, const v_uint16& a, int N = n) {
    auto dh8 = hn::Half<hwy_d_u8>();
    auto d16 = hwy_d_u16();
    auto half = hn::Set(d16, uint16_t(1 << std::max(0, N-1)));
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto max8 = hn::Set(d16, uint16_t(255));
    hn::StoreU(hn::DemoteTo(dh8, hn::Min(as, max8)), dh8, ptr);
}

// i16 → i8 (saturating)
inline v_int8 v_pack(const v_int16& a, const v_int16& b) {
    auto d8  = hwy_d_s8();
    auto d16 = hwy_d_s16();
    auto min8 = hn::Set(d16, int16_t(-128));
    auto max8 = hn::Set(d16, int16_t(127));
    return hn::OrderedDemote2To(d8, hn::Clamp(a, min8, max8), hn::Clamp(b, min8, max8));
}
inline void v_pack_store(schar* ptr, const v_int16& a) {
    auto dh8 = hn::Half<hwy_d_s8>();
    auto d16 = hwy_d_s16();
    auto min8 = hn::Set(d16, int16_t(-128));
    auto max8 = hn::Set(d16, int16_t(127));
    hn::StoreU(hn::DemoteTo(dh8, hn::Clamp(a, min8, max8)), dh8, ptr);
}
template<int n = 0>
inline v_int8 v_rshr_pack(const v_int16& a, const v_int16& b, int N = n) {
    auto d8  = hwy_d_s8();
    auto d16 = hwy_d_s16();
    int16_t half_val = (int16_t)(1 << std::max(0, N-1));
    auto half = hn::Set(d16, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto bs = hn::ShiftRightSame(hn::Add(b, half), N);
    auto min8 = hn::Set(d16, int16_t(-128));
    auto max8 = hn::Set(d16, int16_t(127));
    return hn::OrderedDemote2To(d8, hn::Clamp(as, min8, max8), hn::Clamp(bs, min8, max8));
}
template<int n = 0>
inline void v_rshr_pack_store(schar* ptr, const v_int16& a, int N = n) {
    auto dh8 = hn::Half<hwy_d_s8>();
    auto d16 = hwy_d_s16();
    int16_t half_val = (int16_t)(1 << std::max(0, N-1));
    auto half = hn::Set(d16, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto min8 = hn::Set(d16, int16_t(-128));
    auto max8 = hn::Set(d16, int16_t(127));
    hn::StoreU(hn::DemoteTo(dh8, hn::Clamp(as, min8, max8)), dh8, ptr);
}

// u32 → u16 (saturating)
inline v_uint16 v_pack(const v_uint32& a, const v_uint32& b) {
    auto d16 = hwy_d_u16();
    auto d32 = hwy_d_u32();
    auto max16 = hn::Set(d32, uint32_t(65535));
    return hn::OrderedDemote2To(d16, hn::Min(a, max16), hn::Min(b, max16));
}
inline void v_pack_store(ushort* ptr, const v_uint32& a) {
    auto dh16 = hn::Half<hwy_d_u16>();
    auto d32  = hwy_d_u32();
    auto max16 = hn::Set(d32, uint32_t(65535));
    hn::StoreU(hn::DemoteTo(dh16, hn::Min(a, max16)), dh16, ptr);
}
template<int n = 0>
inline v_uint16 v_rshr_pack(const v_uint32& a, const v_uint32& b, int N = n) {
    auto d16 = hwy_d_u16();
    auto d32  = hwy_d_u32();
    uint32_t half_val = 1u << std::max(0, N-1);
    auto half = hn::Set(d32, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto bs = hn::ShiftRightSame(hn::Add(b, half), N);
    auto max16 = hn::Set(d32, uint32_t(65535));
    return hn::OrderedDemote2To(d16, hn::Min(as, max16), hn::Min(bs, max16));
}
template<int n = 0>
inline void v_rshr_pack_store(ushort* ptr, const v_uint32& a, int N = n) {
    auto dh16 = hn::Half<hwy_d_u16>();
    auto d32  = hwy_d_u32();
    uint32_t half_val = 1u << std::max(0, N-1);
    auto half = hn::Set(d32, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto max16 = hn::Set(d32, uint32_t(65535));
    hn::StoreU(hn::DemoteTo(dh16, hn::Min(as, max16)), dh16, ptr);
}

// i32 → i16 (saturating)
inline v_int16 v_pack(const v_int32& a, const v_int32& b) {
    auto d16 = hwy_d_s16();
    auto d32  = hwy_d_s32();
    auto min16 = hn::Set(d32, int32_t(-32768));
    auto max16 = hn::Set(d32, int32_t(32767));
    return hn::OrderedDemote2To(d16, hn::Clamp(a, min16, max16), hn::Clamp(b, min16, max16));
}
inline void v_pack_store(short* ptr, const v_int32& a) {
    auto dh16 = hn::Half<hwy_d_s16>();
    auto d32  = hwy_d_s32();
    auto min16 = hn::Set(d32, int32_t(-32768));
    auto max16 = hn::Set(d32, int32_t(32767));
    hn::StoreU(hn::DemoteTo(dh16, hn::Clamp(a, min16, max16)), dh16, ptr);
}
template<int n = 0>
inline v_int16 v_rshr_pack(const v_int32& a, const v_int32& b, int N = n) {
    auto d16 = hwy_d_s16();
    auto d32  = hwy_d_s32();
    int32_t half_val = 1 << std::max(0, N-1);
    auto half = hn::Set(d32, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto bs = hn::ShiftRightSame(hn::Add(b, half), N);
    auto min16 = hn::Set(d32, int32_t(-32768));
    auto max16 = hn::Set(d32, int32_t(32767));
    return hn::OrderedDemote2To(d16, hn::Clamp(as, min16, max16), hn::Clamp(bs, min16, max16));
}
template<int n = 0>
inline void v_rshr_pack_store(short* ptr, const v_int32& a, int N = n) {
    auto dh16 = hn::Half<hwy_d_s16>();
    auto d32  = hwy_d_s32();
    int32_t half_val = 1 << std::max(0, N-1);
    auto half = hn::Set(d32, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto min16 = hn::Set(d32, int32_t(-32768));
    auto max16 = hn::Set(d32, int32_t(32767));
    hn::StoreU(hn::DemoteTo(dh16, hn::Clamp(as, min16, max16)), dh16, ptr);
}

// u64 → u32 (wrapping/truncating narrow)
inline v_uint32 v_pack(const v_uint64& a, const v_uint64& b) {
    // Mask to lower 32 bits (wrapping semantics), then demote without saturation
    auto d32  = hwy_d_u32();
    auto d64  = hwy_d_u64();
    auto mask = hn::Set(d64, uint64_t(0xFFFFFFFF));
    auto a32  = hn::And(a, mask);   // values now in [0, UINT32_MAX]
    auto b32  = hn::And(b, mask);
    return hn::OrderedDemote2To(d32, a32, b32);
}
inline void v_pack_store(unsigned* ptr, const v_uint64& a) {
    auto dh32 = hn::Half<hwy_d_u32>();
    auto d64  = hwy_d_u64();
    auto mask = hn::Set(d64, uint64_t(0xFFFFFFFF));
    auto a32  = hn::And(a, mask);
    hn::StoreU(hn::DemoteTo(dh32, a32), dh32, ptr);
}
template<int n = 0>
inline v_uint32 v_rshr_pack(const v_uint64& a, const v_uint64& b, int N = n) {
    auto d32 = hwy_d_u32();
    auto d64 = hwy_d_u64();
    uint64_t half_val = (uint64_t)1 << std::max(0, N-1);
    auto half = hn::Set(d64, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto bs = hn::ShiftRightSame(hn::Add(b, half), N);
    return hn::OrderedDemote2To(d32, as, bs);
}
template<int n = 0>
inline void v_rshr_pack_store(unsigned* ptr, const v_uint64& a, int N = n) {
    auto dh32 = hn::Half<hwy_d_u32>();
    auto d64  = hwy_d_u64();
    uint64_t half_val = (uint64_t)1 << std::max(0, N-1);
    auto half = hn::Set(d64, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    hn::StoreU(hn::DemoteTo(dh32, as), dh32, ptr);
}

// i64 → i32 (wrapping/truncating narrow)
inline v_int32 v_pack(const v_int64& a, const v_int64& b) {
    // Mask lower 32 bits via unsigned path to avoid signed saturation,
    // then reinterpret result as int32 (wrapping semantics)
    auto d32u = hwy_d_u32();
    auto d64u = hn::RebindToUnsigned<hwy_d_s64>();
    auto mask = hn::Set(d64u, uint64_t(0xFFFFFFFF));
    auto a32  = hn::And(v_reinterpret_as_u64(a), mask);  // lower 32 bits in [0, UINT32_MAX]
    auto b32  = hn::And(v_reinterpret_as_u64(b), mask);
    return v_reinterpret_as_s32(hn::OrderedDemote2To(d32u, a32, b32));
}
inline void v_pack_store(int* ptr, const v_int64& a) {
    auto dh32u = hn::Half<hwy_d_u32>();
    auto d64u  = hn::RebindToUnsigned<hwy_d_s64>();
    auto mask  = hn::Set(d64u, uint64_t(0xFFFFFFFF));
    auto a32   = hn::And(v_reinterpret_as_u64(a), mask);
    hn::StoreU(hn::DemoteTo(dh32u, a32), dh32u, reinterpret_cast<unsigned*>(ptr));
}
template<int n = 0>
inline v_int32 v_rshr_pack(const v_int64& a, const v_int64& b, int N = n) {
    auto d32 = hwy_d_s32();
    auto d64 = hwy_d_s64();
    int64_t half_val = (int64_t)1 << std::max(0, N-1);
    auto half = hn::Set(d64, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    auto bs = hn::ShiftRightSame(hn::Add(b, half), N);
    return hn::OrderedDemote2To(d32, as, bs);
}
template<int n = 0>
inline void v_rshr_pack_store(int* ptr, const v_int64& a, int N = n) {
    auto dh32 = hn::Half<hwy_d_s32>();
    auto d64  = hwy_d_s64();
    int64_t half_val = (int64_t)1 << std::max(0, N-1);
    auto half = hn::Set(d64, half_val);
    auto as = hn::ShiftRightSame(hn::Add(a, half), N);
    hn::StoreU(hn::DemoteTo(dh32, as), dh32, ptr);
}

// ── pack_u: signed → unsigned with saturation ────────────────────────────────
// i16 → u8 (clamp to [0, 255])
inline v_uint8 v_pack_u(const v_int16& a, const v_int16& b) {
    auto du8  = hwy_d_u8();
    auto ds16 = hwy_d_s16();
    auto zero = hn::Zero(ds16);
    auto max8 = hn::Set(ds16, int16_t(255));
    auto ac = hn::Clamp(a, zero, max8);
    auto bc = hn::Clamp(b, zero, max8);
    auto du16 = hn::RebindToUnsigned<hwy_d_s16>();
    return hn::OrderedDemote2To(du8, hn::BitCast(du16, ac), hn::BitCast(du16, bc));
}
inline void v_pack_u_store(uchar* ptr, const v_int16& a) {
    auto dhu8 = hn::Half<hwy_d_u8>();
    auto ds16 = hwy_d_s16();
    auto zero = hn::Zero(ds16);
    auto max8 = hn::Set(ds16, int16_t(255));
    auto ac = hn::Clamp(a, zero, max8);
    hn::StoreU(hn::DemoteTo(dhu8, hn::BitCast(hn::RebindToUnsigned<hwy_d_s16>(), ac)), dhu8, ptr);
}
template<int N = 0>
inline v_uint8 v_rshr_pack_u(const v_int16& a, const v_int16& b, int n = N) {
    auto du8  = hwy_d_u8();
    auto ds16 = hwy_d_s16();
    int16_t half_v = (int16_t)(1 << std::max(0, n-1));
    auto half = hn::Set(ds16, half_v);
    auto as = hn::ShiftRightSame(hn::Add(a, half), n);
    auto bs = hn::ShiftRightSame(hn::Add(b, half), n);
    auto zero = hn::Zero(ds16);
    auto max8 = hn::Set(ds16, int16_t(255));
    auto asc = hn::Clamp(as, zero, max8);
    auto bsc = hn::Clamp(bs, zero, max8);
    auto du16 = hn::RebindToUnsigned<hwy_d_s16>();
    return hn::OrderedDemote2To(du8, hn::BitCast(du16, asc), hn::BitCast(du16, bsc));
}
template<int N = 0>
inline void v_rshr_pack_u_store(uchar* ptr, const v_int16& a, int n = N) {
    auto dhu8 = hn::Half<hwy_d_u8>();
    auto ds16 = hwy_d_s16();
    int16_t half_v = (int16_t)(1 << std::max(0, n-1));
    auto half = hn::Set(ds16, half_v);
    auto as = hn::ShiftRightSame(hn::Add(a, half), n);
    auto zero = hn::Zero(ds16);
    auto max8 = hn::Set(ds16, int16_t(255));
    auto asc = hn::Clamp(as, zero, max8);
    auto du16 = hn::RebindToUnsigned<hwy_d_s16>();
    hn::StoreU(hn::DemoteTo(dhu8, hn::BitCast(du16, asc)), dhu8, ptr);
}

// i32 → u16 (clamp to [0, 65535])
inline v_uint16 v_pack_u(const v_int32& a, const v_int32& b) {
    auto du16 = hwy_d_u16();
    auto ds32 = hwy_d_s32();
    auto zero = hn::Zero(ds32);
    auto max16 = hn::Set(ds32, int32_t(65535));
    auto ac = hn::Clamp(a, zero, max16);
    auto bc = hn::Clamp(b, zero, max16);
    auto du32 = hn::RebindToUnsigned<hwy_d_s32>();
    return hn::OrderedDemote2To(du16, hn::BitCast(du32, ac), hn::BitCast(du32, bc));
}
inline void v_pack_u_store(ushort* ptr, const v_int32& a) {
    auto dhu16 = hn::Half<hwy_d_u16>();
    auto ds32  = hwy_d_s32();
    auto zero  = hn::Zero(ds32);
    auto max16 = hn::Set(ds32, int32_t(65535));
    auto ac    = hn::Clamp(a, zero, max16);
    auto du32  = hn::RebindToUnsigned<hwy_d_s32>();
    hn::StoreU(hn::DemoteTo(dhu16, hn::BitCast(du32, ac)), dhu16, ptr);
}
template<int N = 0>
inline v_uint16 v_rshr_pack_u(const v_int32& a, const v_int32& b, int n = N) {
    auto du16 = hwy_d_u16();
    auto ds32 = hwy_d_s32();
    int32_t half_v = 1 << std::max(0, n-1);
    auto half = hn::Set(ds32, half_v);
    auto as = hn::ShiftRightSame(hn::Add(a, half), n);
    auto bs = hn::ShiftRightSame(hn::Add(b, half), n);
    auto zero  = hn::Zero(ds32);
    auto max16 = hn::Set(ds32, int32_t(65535));
    auto asc = hn::Clamp(as, zero, max16);
    auto bsc = hn::Clamp(bs, zero, max16);
    auto du32 = hn::RebindToUnsigned<hwy_d_s32>();
    return hn::OrderedDemote2To(du16, hn::BitCast(du32, asc), hn::BitCast(du32, bsc));
}
template<int N = 0>
inline void v_rshr_pack_u_store(ushort* ptr, const v_int32& a, int n = N) {
    auto dhu16 = hn::Half<hwy_d_u16>();
    auto ds32  = hwy_d_s32();
    int32_t half_v = 1 << std::max(0, n-1);
    auto half = hn::Set(ds32, half_v);
    auto as = hn::ShiftRightSame(hn::Add(a, half), n);
    auto zero  = hn::Zero(ds32);
    auto max16 = hn::Set(ds32, int32_t(65535));
    auto asc = hn::Clamp(as, zero, max16);
    auto du32 = hn::RebindToUnsigned<hwy_d_s32>();
    hn::StoreU(hn::DemoteTo(dhu16, hn::BitCast(du32, asc)), dhu16, ptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Zip  (interleave lower/upper halves)
// ─────────────────────────────────────────────────────────────────────────────
#if HWY_HAVE_SCALABLE
// Scalable targets (RVV/SVE): vectors may span multiple 128-bit blocks.
// Split into half-vectors (each ≤128-bit, single block) and interleave within each half.
// When the half-vector has only 1 lane (e.g. float64 + VLEN=128), fall back to full-vector path.
#define OPENCV_HAL_IMPL_HWY_ZIP(vec_type, d_type)                                         \
inline void v_zip(const vec_type& a, const vec_type& b, vec_type& c, vec_type& d_) {     \
    const d_type d_full;                                                                   \
    const hn::Half<d_type> d_half;                                                        \
    if (hn::Lanes(d_half) <= 1) {                                                         \
        c  = hn::InterleaveLower(d_full, a, b);                                           \
        d_ = hn::InterleaveUpper(d_full, a, b);                                           \
    } else {                                                                               \
        auto a_lo = hn::LowerHalf(d_half, a);                                             \
        auto b_lo = hn::LowerHalf(d_half, b);                                             \
        auto a_hi = hn::UpperHalf(d_half, a);                                             \
        auto b_hi = hn::UpperHalf(d_half, b);                                             \
        c  = hn::Combine(d_full, hn::InterleaveUpper(d_half, a_lo, b_lo),                \
                                 hn::InterleaveLower(d_half, a_lo, b_lo));                \
        d_ = hn::Combine(d_full, hn::InterleaveUpper(d_half, a_hi, b_hi),                \
                                 hn::InterleaveLower(d_half, a_hi, b_hi));                \
    }                                                                                      \
}
#else
// Fixed-width targets (NEON/SSE/AVX2 etc.): vectors are always exactly 128-bit (one block).
// InterleaveLower/Upper on the full vector gives correct semantics directly.
// Half<ScalableTag<uint64_t>> / Half<ScalableTag<double>> are NOT available on NEON,
// so this #else branch avoids instantiating those half-vector types.
#define OPENCV_HAL_IMPL_HWY_ZIP(vec_type, d_type)                                         \
inline void v_zip(const vec_type& a, const vec_type& b, vec_type& c, vec_type& d_) {     \
    const d_type d_full;                                                                   \
    c  = hn::InterleaveLower(d_full, a, b);                                               \
    d_ = hn::InterleaveUpper(d_full, a, b);                                               \
}
#endif

OPENCV_HAL_IMPL_HWY_ZIP(v_uint8,   hwy_d_u8)
OPENCV_HAL_IMPL_HWY_ZIP(v_int8,    hwy_d_s8)
OPENCV_HAL_IMPL_HWY_ZIP(v_uint16,  hwy_d_u16)
OPENCV_HAL_IMPL_HWY_ZIP(v_int16,   hwy_d_s16)
OPENCV_HAL_IMPL_HWY_ZIP(v_uint32,  hwy_d_u32)
OPENCV_HAL_IMPL_HWY_ZIP(v_int32,   hwy_d_s32)
OPENCV_HAL_IMPL_HWY_ZIP(v_float32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_ZIP(v_float64, hwy_d_f64)
#undef OPENCV_HAL_IMPL_HWY_ZIP

// ─────────────────────────────────────────────────────────────────────────────
// Combine  (v_combine_low, v_combine_high, v_recombine)
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_UNPACKS(vec_type, d_type)                                \
inline vec_type v_combine_low(const vec_type& a, const vec_type& b) {                 \
    /* Result = {lower_half(a), lower_half(b)} */                                     \
    auto dh = hn::Half<d_type>();                                                      \
    return hn::Combine(d_type(), hn::LowerHalf(dh, b), hn::LowerHalf(dh, a));       \
}                                                                                      \
inline vec_type v_combine_high(const vec_type& a, const vec_type& b) {                \
    /* Result = {upper_half(a), upper_half(b)} */                                     \
    auto dh = hn::Half<d_type>();                                                      \
    return hn::Combine(d_type(), hn::UpperHalf(dh, b), hn::UpperHalf(dh, a));       \
}                                                                                      \
inline void v_recombine(const vec_type& a, const vec_type& b,                         \
                         vec_type& c, vec_type& d_) {                                  \
    c = v_combine_low(a, b);                                                           \
    d_ = v_combine_high(a, b);                                                         \
}

OPENCV_HAL_IMPL_HWY_UNPACKS(v_uint8,   hwy_d_u8)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_int8,    hwy_d_s8)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_uint16,  hwy_d_u16)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_int16,   hwy_d_s16)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_uint32,  hwy_d_u32)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_int32,   hwy_d_s32)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_float32, hwy_d_f32)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_uint64,  hwy_d_u64)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_int64,   hwy_d_s64)
OPENCV_HAL_IMPL_HWY_UNPACKS(v_float64, hwy_d_f64)
#undef OPENCV_HAL_IMPL_HWY_UNPACKS

// ─────────────────────────────────────────────────────────────────────────────
// Interleaved load / store  (v_load_deinterleave / v_store_interleave)
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_INTERLEAVED(vec_type, lane_tp, d_type)                       \
inline void v_load_deinterleave(const lane_tp* ptr, vec_type& a, vec_type& b) {           \
    hn::LoadInterleaved2(d_type(), ptr, a, b);                                             \
}                                                                                          \
inline void v_load_deinterleave(const lane_tp* ptr, vec_type& a, vec_type& b,            \
                                  vec_type& c) {                                           \
    hn::LoadInterleaved3(d_type(), ptr, a, b, c);                                          \
}                                                                                          \
inline void v_load_deinterleave(const lane_tp* ptr, vec_type& a, vec_type& b,            \
                                  vec_type& c, vec_type& d_) {                             \
    hn::LoadInterleaved4(d_type(), ptr, a, b, c, d_);                                     \
}                                                                                          \
inline void v_store_interleave(lane_tp* ptr, const vec_type& a, const vec_type& b,       \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) {           \
    hn::StoreInterleaved2(a, b, d_type(), ptr);                                           \
}                                                                                          \
inline void v_store_interleave(lane_tp* ptr, const vec_type& a, const vec_type& b,       \
                                const vec_type& c,                                         \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) {           \
    hn::StoreInterleaved3(a, b, c, d_type(), ptr);                                        \
}                                                                                          \
inline void v_store_interleave(lane_tp* ptr, const vec_type& a, const vec_type& b,       \
                                const vec_type& c, const vec_type& d_,                    \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) {           \
    hn::StoreInterleaved4(a, b, c, d_, d_type(), ptr);                                   \
}

OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_uint8,   uchar,    hwy_d_u8)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_int8,    schar,    hwy_d_s8)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_uint16,  ushort,   hwy_d_u16)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_int16,   short,    hwy_d_s16)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_uint32,  unsigned, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_int32,   int,      hwy_d_s32)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_float32, float,    hwy_d_f32)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_uint64,  uint64_t, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_int64,   int64_t,  hwy_d_s64)
OPENCV_HAL_IMPL_HWY_INTERLEAVED(v_float64, double,   hwy_d_f64)
#undef OPENCV_HAL_IMPL_HWY_INTERLEAVED

// ─────────────────────────────────────────────────────────────────────────────
// Interleave pairs/quads  (v_interleave_pairs, v_interleave_quads)
// ─────────────────────────────────────────────────────────────────────────────
// A generic scalar implementation for correctness; platforms can specialise.
#define OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ(vec_type, lane_tp, d_type)                 \
inline vec_type v_interleave_pairs(const vec_type& vec) {                              \
    auto d = d_type();                                                                 \
    int N = VTraits<vec_type>::vlanes();                                               \
    lane_tp buf_in[HWY_MAX_LANES_D(d_type)];                                          \
    lane_tp buf_out[HWY_MAX_LANES_D(d_type)];                                         \
    hn::StoreU(vec, d, buf_in);                                                        \
    for (int i = 0; i < N; i += 4) {                                                  \
        buf_out[i]   = buf_in[i];                                                      \
        buf_out[i+1] = buf_in[i+2];                                                   \
        buf_out[i+2] = buf_in[i+1];                                                   \
        buf_out[i+3] = buf_in[i+3];                                                   \
    }                                                                                  \
    return hn::LoadU(d, buf_out);                                                      \
}                                                                                      \
inline vec_type v_interleave_quads(const vec_type& vec) {                              \
    auto d = d_type();                                                                 \
    int N = VTraits<vec_type>::vlanes();                                               \
    lane_tp buf_in[HWY_MAX_LANES_D(d_type)];                                          \
    lane_tp buf_out[HWY_MAX_LANES_D(d_type)];                                         \
    hn::StoreU(vec, d, buf_in);                                                        \
    for (int i = 0; i < N; i += 8) {                                                  \
        buf_out[i]   = buf_in[i];                                                      \
        buf_out[i+1] = buf_in[i+4];                                                   \
        buf_out[i+2] = buf_in[i+1];                                                   \
        buf_out[i+3] = buf_in[i+5];                                                   \
        buf_out[i+4] = buf_in[i+2];                                                   \
        buf_out[i+5] = buf_in[i+6];                                                   \
        buf_out[i+6] = buf_in[i+3];                                                   \
        buf_out[i+7] = buf_in[i+7];                                                   \
    }                                                                                  \
    return hn::LoadU(d, buf_out);                                                      \
}

OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ(v_uint8,   uint8_t,  hwy_d_u8)
OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ(v_int8,    int8_t,   hwy_d_s8)
OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ(v_uint16,  uint16_t, hwy_d_u16)
OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ(v_int16,   int16_t,  hwy_d_s16)
OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ(v_uint32,  uint32_t, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ(v_int32,   int32_t,  hwy_d_s32)
OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ(v_float32, float,    hwy_d_f32)
#undef OPENCV_HAL_IMPL_HWY_INTERLEAVE_PQ

// ─────────────────────────────────────────────────────────────────────────────
// Horizontal add (v_hadd): adjacent pairs sum to wider type
// ─────────────────────────────────────────────────────────────────────────────
// Template helper: add adjacent pairs producing wider element
#define OPENCV_HAL_IMPL_HWY_HADD_SCALAR(narrow_vec, wide_vec, narrow_tp, wide_tp,  \
                                          narrow_d, wide_d)                          \
static inline wide_vec v_hadd(const narrow_vec& a) {                                 \
    int M = VTraits<wide_vec>::vlanes();                                              \
    narrow_tp buf[HWY_MAX_LANES_D(narrow_d)];                                        \
    wide_tp   out[HWY_MAX_LANES_D(wide_d)];                                           \
    hn::StoreU(a, narrow_d(), buf);                                                   \
    for (int i = 0; i < M; i++)                                                       \
        out[i] = (wide_tp)buf[2*i] + (wide_tp)buf[2*i+1];                            \
    return hn::LoadU(wide_d(), out);                                                  \
}

OPENCV_HAL_IMPL_HWY_HADD_SCALAR(v_uint8,  v_uint16, uint8_t,  uint16_t, hwy_d_u8,  hwy_d_u16)
OPENCV_HAL_IMPL_HWY_HADD_SCALAR(v_uint16, v_uint32, uint16_t, uint32_t, hwy_d_u16, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_HADD_SCALAR(v_uint32, v_uint64, uint32_t, uint64_t, hwy_d_u32, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_HADD_SCALAR(v_int8,   v_int16,  int8_t,   int16_t,  hwy_d_s8,  hwy_d_s16)
OPENCV_HAL_IMPL_HWY_HADD_SCALAR(v_int16,  v_int32,  int16_t,  int32_t,  hwy_d_s16, hwy_d_s32)
OPENCV_HAL_IMPL_HWY_HADD_SCALAR(v_int32,  v_int64,  int32_t,  int64_t,  hwy_d_s32, hwy_d_s64)
#undef OPENCV_HAL_IMPL_HWY_HADD_SCALAR

// ─────────────────────────────────────────────────────────────────────────────
// PopCount
// ─────────────────────────────────────────────────────────────────────────────
static const unsigned char popCountTable_hwy[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

inline v_uint8 v_popcount(const v_uint8& a) {
    return hn::PopulationCount(a);
}
inline v_uint16 v_popcount(const v_uint16& a) {
    return v_hadd(v_popcount(v_reinterpret_as_u8(a)));
}
inline v_uint32 v_popcount(const v_uint32& a) {
    return v_hadd(v_hadd(v_popcount(v_reinterpret_as_u8(a))));
}
inline v_uint64 v_popcount(const v_uint64& a) {
    return v_hadd(v_hadd(v_hadd(v_popcount(v_reinterpret_as_u8(a)))));
}
inline v_uint8  v_popcount(const v_int8&  a) { return v_popcount(v_reinterpret_as_u8(a)); }
inline v_uint16 v_popcount(const v_int16& a) { return v_popcount(v_reinterpret_as_u16(a)); }
inline v_uint32 v_popcount(const v_int32& a) { return v_popcount(v_reinterpret_as_u32(a)); }
inline v_uint64 v_popcount(const v_int64& a) { return v_popcount(v_reinterpret_as_u64(a)); }

// ─────────────────────────────────────────────────────────────────────────────
// SignMask / scan_forward
// ─────────────────────────────────────────────────────────────────────────────
template<typename SvecType, typename DType>
static inline int _hwy_signmask_impl(const SvecType& a, DType d) {
    auto zero = hn::Zero(d);
    auto mask = hn::Lt(a, zero);
    // StoreMaskBits stores one bit per lane; bits[i/8] bit (i%8)
    uint8_t bits[(HWY_MAX_LANES_D(DType) + 7) / 8] = {};
    hn::StoreMaskBits(d, mask, bits);
    int n = VTraits<SvecType>::vlanes();
    int result = 0;
    int bytes = (n + 7) / 8;
    int copy_bytes = std::min((int)sizeof(result), bytes);
    memcpy(&result, bits, (size_t)copy_bytes);
    if (n < 32)
        result &= (1 << n) - 1;
    return result;
}

inline int v_signmask(const v_int8&  a) { return _hwy_signmask_impl(a, hwy_d_s8());  }
inline int v_signmask(const v_int16& a) { return _hwy_signmask_impl(a, hwy_d_s16()); }
inline int v_signmask(const v_int32& a) { return _hwy_signmask_impl(a, hwy_d_s32()); }
inline int v_signmask(const v_int64& a) { return _hwy_signmask_impl(a, hwy_d_s64()); }
inline int64 v_signmask(const v_uint8&  a) { return v_signmask(v_reinterpret_as_s8(a)); }
inline int64 v_signmask(const v_uint16& a) { return v_signmask(v_reinterpret_as_s16(a)); }
inline int   v_signmask(const v_uint32& a) { return v_signmask(v_reinterpret_as_s32(a)); }
inline int   v_signmask(const v_float32& a){ return v_signmask(v_reinterpret_as_s32(a)); }
inline int   v_signmask(const v_uint64& a) { return v_signmask(v_reinterpret_as_s64(a)); }
inline int   v_signmask(const v_float64& a){ return v_signmask(v_reinterpret_as_s64(a)); }

template<typename SvecType, typename DType>
static inline int _hwy_scan_forward_impl(const SvecType& a, DType d) {
    auto zero = hn::Zero(d);
    auto mask = hn::Lt(a, zero);
    intptr_t pos = hn::FindFirstTrue(d, mask);
    return (int)pos;
}

inline int v_scan_forward(const v_int8&  a) { return _hwy_scan_forward_impl(a, hwy_d_s8());  }
inline int v_scan_forward(const v_int16& a) { return _hwy_scan_forward_impl(a, hwy_d_s16()); }
inline int v_scan_forward(const v_int32& a) { return _hwy_scan_forward_impl(a, hwy_d_s32()); }
inline int v_scan_forward(const v_int64& a) { return _hwy_scan_forward_impl(a, hwy_d_s64()); }
inline int v_scan_forward(const v_uint8&  a) { return v_scan_forward(v_reinterpret_as_s8(a)); }
inline int v_scan_forward(const v_uint16& a) { return v_scan_forward(v_reinterpret_as_s16(a)); }
inline int v_scan_forward(const v_uint32& a) { return v_scan_forward(v_reinterpret_as_s32(a)); }
inline int v_scan_forward(const v_float32& a){ return v_scan_forward(v_reinterpret_as_s32(a)); }
inline int v_scan_forward(const v_uint64& a) { return v_scan_forward(v_reinterpret_as_s64(a)); }
inline int v_scan_forward(const v_float64& a){ return v_scan_forward(v_reinterpret_as_s64(a)); }

// ─────────────────────────────────────────────────────────────────────────────
// Pack triplets  {A0,A1,A2,A3, B0,...} → {A0,A1,A2, B0,...}
// ─────────────────────────────────────────────────────────────────────────────
#define OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(vec_type, lane_tp, d_type)           \
inline vec_type v_pack_triplets(const vec_type& vec) {                          \
    auto d = d_type();                                                          \
    int N = VTraits<vec_type>::vlanes();                                        \
    lane_tp buf_in[HWY_MAX_LANES_D(d_type)];                                   \
    lane_tp buf_out[HWY_MAX_LANES_D(d_type)];                                  \
    hn::StoreU(vec, d, buf_in);                                                 \
    int out_i = 0;                                                              \
    for (int i = 0; i < N && out_i < N; i++) {                                 \
        if ((i & 3) != 3)  /* skip every 4th element */                        \
            buf_out[out_i++] = buf_in[i];                                       \
    }                                                                           \
    /* fill remainder with zeros if N not divisible by 4 */                    \
    while (out_i < N) buf_out[out_i++] = (lane_tp)0;                           \
    return hn::LoadU(d, buf_out);                                               \
}

OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_uint8,   uint8_t,  hwy_d_u8)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_int8,    int8_t,   hwy_d_s8)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_uint16,  uint16_t, hwy_d_u16)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_int16,   int16_t,  hwy_d_s16)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_uint32,  uint32_t, hwy_d_u32)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_int32,   int32_t,  hwy_d_s32)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_float32, float,    hwy_d_f32)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_uint64,  uint64_t, hwy_d_u64)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_int64,   int64_t,  hwy_d_s64)
OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS(v_float64, double,   hwy_d_f64)
#undef OPENCV_HAL_IMPL_HWY_PACK_TRIPLETS

// ─────────────────────────────────────────────────────────────────────────────
// FP16 support (hfloat = 16-bit float, stored as uint16/ushort in OpenCV)
// ─────────────────────────────────────────────────────────────────────────────
inline v_float32 v_load_expand(const hfloat* ptr) {
    auto d32 = hwy_d_f32();
    int N = VTraits<v_float32>::vlanes();
    float buf[HWY_MAX_LANES_D(hwy_d_f32)];
    for (int i = 0; i < N; i++)
        buf[i] = (float)ptr[i];
    return hn::LoadU(d32, buf);
}

inline void v_pack_store(hfloat* ptr, const v_float32& v) {
    int N = VTraits<v_float32>::vlanes();
    float buf[HWY_MAX_LANES_D(hwy_d_f32)];
    hn::StoreU(v, hwy_d_f32(), buf);
    for (int i = 0; i < N; i++)
        ptr[i] = hfloat(buf[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Rounding
// ─────────────────────────────────────────────────────────────────────────────
inline v_int32 v_round(const v_float32& a) {
    // Highway NearestInt converts float → int with round-to-nearest
    return hn::NearestInt(a);
}
inline v_int32 v_floor(const v_float32& a) {
    return hn::ConvertTo(hwy_d_s32(), hn::Floor(a));
}
inline v_int32 v_ceil(const v_float32& a) {
    return hn::ConvertTo(hwy_d_s32(), hn::Ceil(a));
}
inline v_int32 v_trunc(const v_float32& a) {
    // TruncateTo is for integer narrowing; use ConvertTo+Trunc for float→int.
    return hn::ConvertTo(hwy_d_s32(), hn::Trunc(a));
}

inline v_int32 v_round(const v_float64& a) {
    // a has N/2 f64 lanes. NearestInt → N/2 i64. OrderedDemote2To needs two
    // equal-width i64 inputs; use Zero(hwy_d_s64()) for the second half.
    auto di32    = hwy_d_s32();
    auto di64    = hwy_d_s64();              // N/2 lanes (same as f64)
    auto rounded = hn::NearestInt(a);        // N/2 int64
    auto zeros   = hn::Zero(di64);           // N/2 zeros
    return hn::OrderedDemote2To(di32, rounded, zeros);
}

inline v_int32 v_round(const v_float64& a, const v_float64& b) {
    auto di32 = hwy_d_s32();
    auto r_a  = hn::NearestInt(a);           // N/2 int64
    auto r_b  = hn::NearestInt(b);           // N/2 int64
    return hn::OrderedDemote2To(di32, r_a, r_b);
}

inline v_int32 v_floor(const v_float64& a) {
    auto di32  = hwy_d_s32();
    auto di64  = hwy_d_s64();
    auto zeros = hn::Zero(di64);
    auto fi    = hn::ConvertTo(di64, hn::Floor(a));
    return hn::OrderedDemote2To(di32, fi, zeros);
}

inline v_int32 v_ceil(const v_float64& a) {
    auto di32  = hwy_d_s32();
    auto di64  = hwy_d_s64();
    auto zeros = hn::Zero(di64);
    auto fi    = hn::ConvertTo(di64, hn::Ceil(a));
    return hn::OrderedDemote2To(di32, fi, zeros);
}

inline v_int32 v_trunc(const v_float64& a) {
    auto di32  = hwy_d_s32();
    auto di64  = hwy_d_s64();
    auto zeros = hn::Zero(di64);
    auto fi    = hn::ConvertTo(di64, hn::Trunc(a));
    return hn::OrderedDemote2To(di32, fi, zeros);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dot product
// ─────────────────────────────────────────────────────────────────────────────
// 16 → 32  (pairwise: result[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1])
inline v_int32 v_dotprod(const v_int16& a, const v_int16& b) {
    return hn::WidenMulPairwiseAdd(hwy_d_s32(), a, b);
}
inline v_int32 v_dotprod(const v_int16& a, const v_int16& b, const v_int32& c) {
    return hn::Add(v_dotprod(a, b), c);
}

// 32 → 64  (pairwise: result[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1])
inline v_int64 v_dotprod(const v_int32& a, const v_int32& b) {
    // MulEven/MulOdd: widening multiply of even/odd lanes
    return hn::Add(hn::MulEven(a, b), hn::MulOdd(a, b));
}
inline v_int64 v_dotprod(const v_int32& a, const v_int32& b, const v_int64& c) {
    return hn::Add(v_dotprod(a, b), c);
}

// 8 → 32  (expand factor 4)
inline v_uint32 v_dotprod_expand(const v_uint8& a, const v_uint8& b) {
    auto d32 = hwy_d_u32();
    uint8_t  ba[HWY_MAX_LANES_D(hwy_d_u8)];
    uint8_t  bb[HWY_MAX_LANES_D(hwy_d_u8)];
    uint32_t out[HWY_MAX_LANES_D(hwy_d_u32)] = {};
    hn::StoreU(a, hwy_d_u8(), ba);
    hn::StoreU(b, hwy_d_u8(), bb);
    int N8 = VTraits<v_uint8>::vlanes();
    for (int i = 0; i < N8; i++) out[i / 4] += (uint32_t)ba[i] * (uint32_t)bb[i];
    return hn::LoadU(d32, out);
}
inline v_uint32 v_dotprod_expand(const v_uint8& a, const v_uint8& b, const v_uint32& c) {
    return hn::Add(v_dotprod_expand(a, b), c);
}
inline v_int32 v_dotprod_expand(const v_int8& a, const v_int8& b) {
    auto d32 = hwy_d_s32();
    int N8 = VTraits<v_int8>::vlanes();
    int8_t  ba[HWY_MAX_LANES_D(hwy_d_s8)];
    int8_t  bb[HWY_MAX_LANES_D(hwy_d_s8)];
    int32_t out[HWY_MAX_LANES_D(hwy_d_s32)] = {};
    hn::StoreU(a, hwy_d_s8(), ba);
    hn::StoreU(b, hwy_d_s8(), bb);
    for (int i = 0; i < N8; i++) out[i / 4] += (int32_t)ba[i] * (int32_t)bb[i];
    return hn::LoadU(d32, out);
}
inline v_int32 v_dotprod_expand(const v_int8& a, const v_int8& b, const v_int32& c) {
    return hn::Add(v_dotprod_expand(a, b), c);
}

// 16 → 64
inline v_uint64 v_dotprod_expand(const v_uint16& a, const v_uint16& b) {
    // Use MulEven: u32 results, then widen-add adjacent pairs
    auto d64 = hwy_d_u64();
    int N16 = VTraits<v_uint16>::vlanes();
    uint16_t ba[HWY_MAX_LANES_D(hwy_d_u16)];
    uint16_t bb[HWY_MAX_LANES_D(hwy_d_u16)];
    uint64_t out[HWY_MAX_LANES_D(hwy_d_u64)] = {};
    hn::StoreU(a, hwy_d_u16(), ba);
    hn::StoreU(b, hwy_d_u16(), bb);
    for (int i = 0; i < N16; i++) out[i / 4] += (uint64_t)ba[i] * (uint64_t)bb[i];
    return hn::LoadU(d64, out);
}
inline v_uint64 v_dotprod_expand(const v_uint16& a, const v_uint16& b, const v_uint64& c) {
    return hn::Add(v_dotprod_expand(a, b), c);
}
inline v_int64 v_dotprod_expand(const v_int16& a, const v_int16& b) {
    auto d64 = hwy_d_s64();
    int N16 = VTraits<v_int16>::vlanes();
    int16_t ba[HWY_MAX_LANES_D(hwy_d_s16)];
    int16_t bb[HWY_MAX_LANES_D(hwy_d_s16)];
    int64_t out[HWY_MAX_LANES_D(hwy_d_s64)] = {};
    hn::StoreU(a, hwy_d_s16(), ba);
    hn::StoreU(b, hwy_d_s16(), bb);
    for (int i = 0; i < N16; i++) out[i / 4] += (int64_t)ba[i] * (int64_t)bb[i];
    return hn::LoadU(d64, out);
}
inline v_int64 v_dotprod_expand(const v_int16& a, const v_int16& b, const v_int64& c) {
    return hn::Add(v_dotprod_expand(a, b), c);
}

inline v_float64 v_dotprod_expand(const v_int32& a, const v_int32& b) {
    return v_cvt_f64(v_dotprod(a, b));
}
inline v_float64 v_dotprod_expand(const v_int32& a, const v_int32& b, const v_float64& c) {
    return hn::Add(v_dotprod_expand(a, b), c);
}

// ── Fast dot product (reduces into scalar, stored in first element) ──────────
inline v_int32 v_dotprod_fast(const v_int16& a, const v_int16& b) {
    int32_t s = (int32_t)hn::ReduceSum(hwy_d_s32(), v_dotprod(a, b));
    auto r = hn::Zero(hwy_d_s32());
    return hn::InsertLane(r, 0, s);
}
inline v_int32 v_dotprod_fast(const v_int16& a, const v_int16& b, const v_int32& c) {
    return hn::Add(c, v_dotprod_fast(a, b));
}
inline v_int64 v_dotprod_fast(const v_int32& a, const v_int32& b) {
    int64_t s = (int64_t)hn::ReduceSum(hwy_d_s64(), v_dotprod(a, b));
    auto r = hn::Zero(hwy_d_s64());
    return hn::InsertLane(r, 0, s);
}
inline v_int64 v_dotprod_fast(const v_int32& a, const v_int32& b, const v_int64& c) {
    return hn::Add(c, v_dotprod_fast(a, b));
}
inline v_uint32 v_dotprod_expand_fast(const v_uint8& a, const v_uint8& b) {
    uint64_t s = (uint64_t)hn::ReduceSum(hwy_d_u32(), v_dotprod_expand(a, b));
    auto r = hn::Zero(hwy_d_u32());
    return hn::InsertLane(r, 0, (uint32_t)s);
}
inline v_uint32 v_dotprod_expand_fast(const v_uint8& a, const v_uint8& b, const v_uint32& c) {
    return hn::Add(c, v_dotprod_expand_fast(a, b));
}
inline v_int32 v_dotprod_expand_fast(const v_int8& a, const v_int8& b) {
    int64_t s = (int64_t)hn::ReduceSum(hwy_d_s32(), v_dotprod_expand(a, b));
    auto r = hn::Zero(hwy_d_s32());
    return hn::InsertLane(r, 0, (int32_t)s);
}
inline v_int32 v_dotprod_expand_fast(const v_int8& a, const v_int8& b, const v_int32& c) {
    return hn::Add(c, v_dotprod_expand_fast(a, b));
}
inline v_uint64 v_dotprod_expand_fast(const v_uint16& a, const v_uint16& b) {
    uint64_t s = hn::ReduceSum(hwy_d_u64(), v_dotprod_expand(a, b));
    auto r = hn::Zero(hwy_d_u64());
    return hn::InsertLane(r, 0, s);
}
inline v_uint64 v_dotprod_expand_fast(const v_uint16& a, const v_uint16& b, const v_uint64& c) {
    return hn::Add(c, v_dotprod_expand_fast(a, b));
}
inline v_int64 v_dotprod_expand_fast(const v_int16& a, const v_int16& b) {
    int64_t s = hn::ReduceSum(hwy_d_s64(), v_dotprod_expand(a, b));
    auto r = hn::Zero(hwy_d_s64());
    return hn::InsertLane(r, 0, s);
}
inline v_int64 v_dotprod_expand_fast(const v_int16& a, const v_int16& b, const v_int64& c) {
    return hn::Add(c, v_dotprod_expand_fast(a, b));
}
inline v_float64 v_dotprod_expand_fast(const v_int32& a, const v_int32& b) {
    double s = hn::ReduceSum(hwy_d_f64(), v_dotprod_expand(a, b));
    auto r = hn::Zero(hwy_d_f64());
    return hn::InsertLane(r, 0, s);
}
inline v_float64 v_dotprod_expand_fast(const v_int32& a, const v_int32& b, const v_float64& c) {
    return hn::Add(c, v_dotprod_expand_fast(a, b));
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix multiply (4-element, first 4 lanes only)
// ─────────────────────────────────────────────────────────────────────────────
inline v_float32 v_matmul(const v_float32& v, const v_float32& mat0,
                            const v_float32& mat1, const v_float32& mat2,
                            const v_float32& mat3) {
    float e0 = hn::ExtractLane(v, 0);
    float e1 = hn::ExtractLane(v, 1);
    float e2 = hn::ExtractLane(v, 2);
    float e3 = hn::ExtractLane(v, 3);
    auto df = hwy_d_f32();
    return hn::Add(hn::Add(hn::Mul(mat0, hn::Set(df, e0)),
                                 hn::Mul(mat1, hn::Set(df, e1))),
                   hn::Add(hn::Mul(mat2, hn::Set(df, e2)),
                                 hn::Mul(mat3, hn::Set(df, e3))));
}

inline v_float32 v_matmuladd(const v_float32& v, const v_float32& mat0,
                               const v_float32& mat1, const v_float32& mat2,
                               const v_float32& a) {
    float e0 = hn::ExtractLane(v, 0);
    float e1 = hn::ExtractLane(v, 1);
    float e2 = hn::ExtractLane(v, 2);
    auto df = hwy_d_f32();
    return hn::Add(hn::Add(hn::Mul(mat0, hn::Set(df, e0)),
                                 hn::Mul(mat1, hn::Set(df, e1))),
                   hn::Add(hn::Mul(mat2, hn::Set(df, e2)), a));
}

// ─────────────────────────────────────────────────────────────────────────────
// v_cleanup
// ─────────────────────────────────────────────────────────────────────────────
inline void v_cleanup() {}

// ─────────────────────────────────────────────────────────────────────────────
// Math functions via generic default implementations
// ─────────────────────────────────────────────────────────────────────────────
#include "intrin_math.hpp"

inline v_float32 v_exp(const v_float32& x)   { return v_exp_default_32f<v_float32, v_int32>(x); }
inline v_float32 v_log(const v_float32& x)   { return v_log_default_32f<v_float32, v_int32>(x); }
inline v_float32 v_erf(const v_float32& x)   { return v_erf_default_32f<v_float32, v_int32>(x); }
inline void v_sincos(const v_float32& x, v_float32& s, v_float32& c) {
    v_sincos_default_32f<v_float32, v_int32>(x, s, c);
}
inline v_float32 v_sin(const v_float32& x) { return v_sin_default_32f<v_float32, v_int32>(x); }
inline v_float32 v_cos(const v_float32& x) { return v_cos_default_32f<v_float32, v_int32>(x); }

inline v_float64 v_exp(const v_float64& x)   { return v_exp_default_64f<v_float64, v_int64>(x); }
inline v_float64 v_log(const v_float64& x)   { return v_log_default_64f<v_float64, v_int64>(x); }
inline void v_sincos(const v_float64& x, v_float64& s, v_float64& c) {
    v_sincos_default_64f<v_float64, v_int64>(x, s, c);
}
inline v_float64 v_sin(const v_float64& x) { return v_sin_default_64f<v_float64, v_int64>(x); }
inline v_float64 v_cos(const v_float64& x) { return v_cos_default_64f<v_float64, v_int64>(x); }

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // namespace cv

#endif // OPENCV_HAL_INTRIN_HIGHWAY_HPP