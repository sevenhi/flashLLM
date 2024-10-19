#include <iostream>
#include "src/add/add.hpp"
#include "fmt/core.h"

int main()
{
    int x = 3;
    int y = 4;
    int out = add(x, y);

    fmt::print("out is: {}\n", out);
}