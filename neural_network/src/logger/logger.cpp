#include "logger.h"
#include <ctime>
#include <iomanip>
#include <chrono>

using namespace std;

void Logger::printCurrentTime(ostream &os) {
    std::time_t timenow = std::time(nullptr);
    os << std::put_time(std::localtime(&timenow), "%Y/%m/%d %T");
}

void Logger::logWithLevel(const string &level, const string &msg, ostream &os) {
    os << "[";
    printCurrentTime(os);
    os << "] " << "[" << level << "] - " << msg << '\n';
}

void Logger::info(const string &msg, ostream &os /* = std::cout */) {
    logWithLevel(INFO, msg, os);
}

void Logger::error(const string &msg, ostream &os /* = std::cout */) {
    logWithLevel(ERROR, msg, os);
}

void Logger::info(float num, ostream &os /* = std::cout */) {
    Logger::info(to_string(num), os);
}

void Logger::measureTime(void (*f)(), ostream &os /* = std::cout */) {
    typedef chrono::steady_clock::time_point time_point;

    time_point begin = std::chrono::steady_clock::now();
    f();
    time_point end = std::chrono::steady_clock::now();

    Logger::info("Execution took " + to_string(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) + "[µs]", os);
}
