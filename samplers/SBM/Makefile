CC = g++
CFLAGS = -std=c++17 -w -O2

SBM.exe:SBM.o
	$(CC) $(CFLAGS) -o SBM.exe SBM.o $(LIBS)

SBM.o:SBM.cpp Makefile
	$(CC) $(CFLAGS) -c SBM.cpp 

