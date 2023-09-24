CC=gcc
CFLAGS=-Wall -Werror -I.
all: xor

xor : xor.o 
	@echo Linking...
	$(CC) $(CFLAGS) -o xor xor.o -lm
	@echo Done!

main.o: xor.c
	@echo Compiling...
	$(CC) $(CFLAGS) -c main.c


clean:
	$(RM) xor *.o *~
	@echo All clean!