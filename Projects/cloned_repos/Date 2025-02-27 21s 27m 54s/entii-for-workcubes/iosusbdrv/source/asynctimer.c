#define DEVL 1
#include <ntddk.h>
#include "asynctimer.h"

#define MS_TO_TIMEOUT(ms) ((ms) * 10000)

void AsyncTimerSet(PKTIMER Timer, PRKDPC Dpc) {
	LARGE_INTEGER DueTime;
	DueTime.QuadPart = -MS_TO_TIMEOUT(10);
	KeSetTimer(Timer, DueTime, Dpc);
}