#ifndef NEURAL_NETWORK_LOGGER_H
#define NEURAL_NETWORK_LOGGER_H

#include <vector>
#include <iostream>

using namespace std;

class Logger {
    inline static const string INFO = "INFO";
    inline static const string ERROR = "ERROR";
    static void printCurrentTime(ostream &os);
    static void logWithLevel(const string& level, const string& msg, ostream& os);
public:
    static void info(const string& msg, ostream& os = cout);
    static void info(float num, ostream& os = cout);
    static void error(const string& msg, ostream& os = cout);

    static void measureTime(void (*f)(), ostream &os = cout);
};


#endif //NEURAL_NETWORK_LOGGER_H
