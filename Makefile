CC=g++
RM=rm
CFLAGS=-Iinclude -std=c++20 -Xclang -fopenmp -lomp

all: perms

perms: src/perms.cpp include/config.h
	$(CC) $(CFLAGS) -O3 src/perms.cpp -o perms

debug: src/perms.cpp include/config.h
	$(CC) $(CFLAGS) -g -Og -Wall -pedantic src/perms.cpp -o perms

clean:
	$(RM) perms

