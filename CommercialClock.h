//
// Created by ktoztam on 20.06.2021.
//

#ifndef COMMERCIALDETECTOR_COMMERCIALCLOCK_H
#define COMMERCIALDETECTOR_COMMERCIALCLOCK_H

#include <iostream>
#include <vector>
#include <tuple>
#include <sstream>
#include <iomanip>

#define COMMERCIAL true
#define MOVIE false

// Pomocniczno z https://stackoverflow.com/questions/58695875/how-to-convert-seconds-to-hhmmss-millisecond-format-c
std::string format_time(float seconds) {
    int ms = (int) (seconds * 1000);
    int h = ms / (1000 * 60 * 60);
    ms -= h * (1000 * 60 * 60);
    int m = ms / (1000 * 60);
    ms -= m * (1000 * 60);
    int s = ms / 1000;
    ms -= s * 1000;
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << h << ':' << std::setw(2) << m
       << ':' << std::setw(2) << s << '.' << std::setw(3) << ms;
    return ss.str();
}

class CommercialClock {
protected:
    int fps;
    std::vector<std::tuple<int, bool>> timestamps;
public:
    CommercialClock(int fps) : fps(fps) {}

    void registerType(bool isCommercial, int frame) {
        if (frame == 0) {
            timestamps.emplace_back(std::make_tuple(frame, isCommercial));
        } else {
            auto[last_frame, last_type] = timestamps.back();
            if (last_type != isCommercial) {
                if (frame - last_frame < fps * 5) {
                    timestamps.pop_back();
                } else {
                    timestamps.emplace_back(std::make_tuple(frame, isCommercial));
                }

            }
        }
    }

    std::string statistics() {
        std::stringstream stream;
        stream << "Statistics for this clip: \n";
        for (int i = 0; i < timestamps.size() - 1; i++) {
            auto[frame, type] = timestamps[i];
            auto[next_frame, _] = timestamps[i + 1];
            stream << (type ? "Commmercial: " : "Movie: ") << format_time((1.f*frame)/fps) << " - " << format_time((1.f*next_frame)/fps) << std::endl;
        }
        return stream.str();
    }

    void finish(int frameIdx) {
        auto[_, lastType] = timestamps.back();
        timestamps.emplace_back(std::make_tuple(frameIdx, lastType));
    }

};


#endif //COMMERCIALDETECTOR_COMMERCIALCLOCK_H
