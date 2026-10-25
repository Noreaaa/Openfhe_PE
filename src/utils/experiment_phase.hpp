#pragma once
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

// Sampling marker only: no process high-water reset and no allocator intervention.
inline void WriteExperimentPhase(const std::string& profile, const std::string& phase, int layer=0) {
    if(profile.empty())return;
    const auto path=profile+".phase.json",temp=path+".tmp";
    const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    {
        std::ofstream out(temp);
        if(!out)throw std::runtime_error("cannot write experiment phase");
        out<<"{\"phase\":\""<<phase<<"\",\"layer\":"<<layer<<",\"unix_ms\":"<<ms<<"}\n";
    }
    if(std::rename(temp.c_str(),path.c_str()))throw std::runtime_error("cannot publish experiment phase");
}
