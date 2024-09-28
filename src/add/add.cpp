#include "add.hpp"

template <typename T>
T add(T x, T y)
{
    return x + y;
}

template int add(int, int);
template float add(float, float);
