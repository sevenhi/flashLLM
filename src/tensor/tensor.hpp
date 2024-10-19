#include <vector>
#include <stdio.h>

template <typename T>
class Tensor
{
private:
    void *data;
    std::vector<size_t> shape;
    std::vector<size_t> stride;

    Tensor() = default;
};