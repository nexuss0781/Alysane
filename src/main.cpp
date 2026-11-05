#include "alysane.h"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: alysane <selftest|unittest|parity|gates|complete> ...\n";
            return 2;
        }
        std::string cmd = argv[1];
        if (cmd == "selftest") {
            alysane::run_selftest();
            return 0;
        }
        if (cmd == "unittest") {
            alysane::run_unit_tests();
            return 0;
        }
        if (cmd == "parity" && argc >= 5) {
            alysane::run_parity(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (cmd == "gates" && argc >= 5) {
            alysane::run_gates(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (cmd == "complete" && argc >= 5) {
            alysane::run_completion_gates(argv[2], argv[3], argv[4]);
            return 0;
        }
        std::cerr << "usage: alysane selftest | unittest | parity MODEL TOKENS OUT | gates MODEL DATA OUT | "
                     "complete MODEL DATA OUT\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
