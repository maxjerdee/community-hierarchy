CC = g++
CFLAGS = -std=c++17 -w -O2

SBM:SBM.o
	$(CC) $(CFLAGS) -o SBM SBM.o $(LIBS)

SBM.o:SBM.cpp Makefile
	$(CC) $(CFLAGS) -c SBM.cpp 

