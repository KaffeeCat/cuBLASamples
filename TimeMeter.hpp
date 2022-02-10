/////////////////////////////////////////
// Filename: TimeMeter.hpp
// ------------------------------------------------
// Purpose: Time meter to measure time usage (ms)
// Author: Wang Kang
// Date: 2022/2/10
/////////////////////////////////////////
#pragma once

#include <windows.h>

class TimeMeter
{
public:
	TimeMeter() { QueryPerformanceFrequency(&_freq); };
	~TimeMeter() {}

	void start() { QueryPerformanceCounter(&_start); }
	double stop()
	{
		LARGE_INTEGER end;
		QueryPerformanceCounter(&end); 
		return 1000 * ((double)end.QuadPart - (double)_start.QuadPart) / (double)_freq.QuadPart;
	}

private:
	LARGE_INTEGER _start;
	LARGE_INTEGER _freq;
};
