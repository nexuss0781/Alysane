#include "alysane.h"
#include <iostream>

int main() {
    try {
        alysane::run_unit_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
