#include <stdio.h>
#include <stdlib.h>

struct timespec getDiffTime(struct timespec *startTime, struct timespec *endTime)
{
	struct timespec diff;                                                                         
	diff.tv_sec = endTime->tv_sec - startTime->tv_sec;                                            
	diff.tv_nsec = endTime->tv_nsec - startTime->tv_nsec;                                         
	if (diff.tv_nsec < 0) {                                                                       
		diff.tv_nsec += 1.0e9;                                                                
		diff.tv_sec--;                                                                        
	}                                                                                             
	return diff;                                                                                  
} 
