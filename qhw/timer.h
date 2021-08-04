#include "cholmod.h"

static double GetTime() {
	return double(SuiteSparse_time()); 
}