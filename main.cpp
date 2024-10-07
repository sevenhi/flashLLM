#include <iostream>
#include "src/add/add.hpp"
#include "fmt/format.h"

int main()
{
    int x = 3;
    int y = 4;
    int out = add(x, y);

    fmt::print("out is:\n {}\n", out);
}