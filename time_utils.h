#pragma once
#include <string>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

class TimerLogger {
public:
    static std::string getCurrentTimeStr() {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
        return ss.str();
    }

    static void logStep(const char* step_name, const char* detail = nullptr) {
        if (detail) {
            printf("[%s] %s: %s\n", getCurrentTimeStr().c_str(), step_name, detail);
        } else {
            printf("[%s] %s\n", getCurrentTimeStr().c_str(), step_name);
        }
    }

    static void logDuration(const char* step_name, double duration_ms) {
        printf("[%s] %s took %.2f ms\n", getCurrentTimeStr().c_str(), step_name, duration_ms);
    }

    class ScopedTimer {
    public:
        ScopedTimer(const char* name) : name_(name), start_(std::chrono::steady_clock::now()) {}
        
        ~ScopedTimer() {
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
            logDuration(name_, duration / 1000.0);
        }

    private:
        const char* name_;
        std::chrono::steady_clock::time_point start_;
    };
};

#define TIME_LOG(name) TimerLogger::ScopedTimer timer##__LINE__(name)
#define LOG_STEP(step) TimerLogger::logStep(step)
#define LOG_STEP_DETAIL(step, detail) TimerLogger::logStep(step, detail)
