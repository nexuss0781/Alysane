#include "c3v2_1.h"
#include <iostream>
int main(int argc, char** argv) {
    try {
        std::string out = argc > 1 ? argv[1] : "reports/c3v2_1_demo.json";
        alysane::c3v2_1::run_checks();
        alysane::c3v2_1::write_demo_report(out);
        std::cout << "C3v2-1 checks passed. Report: " << out << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "C3v2-1 failure: " << e.what() << "\n";
        return 1;
    }
}
