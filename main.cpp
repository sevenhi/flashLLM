#include <iostream>
#include "src/add/add.hpp"

int main()
{
    int x = 3;
    int y = 4;
    int out = add(x, y);

    std::cout << out << std::endl;
}