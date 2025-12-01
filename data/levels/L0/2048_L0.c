

#define VERSION "1.0.3"

#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>

#define SIZE 4


void getColors(uint8_t value, uint8_t scheme, uint8_t *foreground, uint8_t *background)
{


}

uint8_t getDigitCount(uint32_t number)
{

}

void drawBoard(uint8_t board[SIZE][SIZE], uint8_t scheme, uint32_t score)
{

}

uint8_t findTarget(uint8_t array[SIZE], uint8_t x, uint8_t stop)
{

}

bool slideArray(uint8_t array[SIZE], uint32_t *score)
{

}

void rotateBoard(uint8_t board[SIZE][SIZE])
{

}

bool moveUp(uint8_t board[SIZE][SIZE], uint32_t *score)
{

}

bool moveLeft(uint8_t board[SIZE][SIZE], uint32_t *score)
{

}

bool moveDown(uint8_t board[SIZE][SIZE], uint32_t *score)
{

}

bool moveRight(uint8_t board[SIZE][SIZE], uint32_t *score)
{

}

bool findPairDown(uint8_t board[SIZE][SIZE])
{

}

uint8_t countEmpty(uint8_t board[SIZE][SIZE])
{

}

bool gameEnded(uint8_t board[SIZE][SIZE])
{

}

void addRandom(uint8_t board[SIZE][SIZE])
{

}

void initBoard(uint8_t board[SIZE][SIZE])
{

}

void setBufferedInput(bool enable)
{

}

bool testSucceed()
{

}

void signal_callback_handler(int signum)
{

}

int main(int argc, char *argv[])
{

}
