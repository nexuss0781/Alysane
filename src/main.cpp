#include "alysane.h"
#include "probe.h"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: alysane <selftest|unittest|parity|gates|complete|probe> ...\n";
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
        if (cmd == "probe" && argc >= 5) {
            alysane::probe::Config cfg;
            if (const char* e = std::getenv("ALYSANE_PROBE_GRID")) cfg.eps_grid = std::atoi(e);
            if (const char* e = std::getenv("ALYSANE_PROBE_B")) cfg.bootstrap_B = std::atoi(e);
            if (const char* e = std::getenv("ALYSANE_PROBE_M")) cfg.M_pairs = std::atoi(e);
            if (const char* e = std::getenv("ALYSANE_PROBE_LANDMARKS")) cfg.landmark_m = std::atoi(e);
            if (const char* e = std::getenv("ALYSANE_DELTA_BUDGET")) cfg.delta_budget = (float)std::atof(e);
            auto r = alysane::probe::run(argv[2], argv[3], argv[4], cfg);
            // Exit non-zero on NO_GO so CI surfaces the pre-committed branch decision.
            return r.final_verdict == "NO_GO" ? 3 : 0;
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
