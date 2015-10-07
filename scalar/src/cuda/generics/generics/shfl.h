#pragma once
#include <generics/detail/alias.h>
#include <thrust/detail/static_assert.h>

namespace detail {

template<int s>
struct shuffle {
    __device__ __forceinline__
    static void impl(array<int, s>& d, const int& i) {
        d.head = __shfl(d.head, i);
        shuffle<s-1>::impl(d.tail, i);
    }
};

template<>
struct shuffle<1> {
    __device__ __forceinline__
    static void impl(array<int, 1>& d, const int& i) {
        d.head = __shfl(d.head, i);
    }
};

}

template<typename T>
__device__ __forceinline__
T __shfl(const T& t, const int& i) {
    
    //X If you get a compiler error on this line, it is because
    //X sizeof(T) is not divisible by 4, and so this type is not
    //X supported currently.
    THRUST_STATIC_ASSERT((detail::size_multiple_power_of_two<T, 2>::value));

    
    typedef typename detail::working_array<T>::type aliased;
    aliased lysed = detail::lyse<int>(t);
    detail::shuffle<aliased::size>::impl(lysed, i);
    return detail::fuse<T>(lysed);
}
