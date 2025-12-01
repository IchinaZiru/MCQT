/*
 ============================================================================
 Name        : 2048.c
 Author      : Maurits van der Schee
 Description : Console version of the game "2048" for GNU/Linux
 ============================================================================
 */

#define VERSION "1.0.3"

#define _XOPEN_SOURCE 500 // for: usleep
#include <stdio.h>		  // defines: printf, puts, getchar
#include <stdlib.h>		  // defines: EXIT_SUCCESS
#include <string.h>		  // defines: strcmp
#include <unistd.h>		  // defines: STDIN_FILENO, usleep
#include <termios.h>	  // defines: termios, TCSANOW, ICANON, ECHO
#include <stdbool.h>	  // defines: true, false
#include <stdint.h>		  // defines: uint8_t, uint32_t
#include <time.h>		  // defines: time
#include <signal.h>		  // defines: signal, SIGINT

#define SIZE 4

// this function receives 2 pointers (indicated by *) so it can set their values
/* @doc @function getColors @level L4
 * @logical この関数は、指定されたカラースキームに基づいて前景色と背景色を取得します。
 * @precise 指定されたスキーム配列から、valueに基づいて前景色と背景色を計算してポインタに格納します。
 * @unambiguous スキーム配列のインデックスを計算し、前景色と背景色のポインタに対応する色を設定します。
 * @exhaustive 戻り値はなく、指定されたポインタに前景色と背景色を設定します。スキームが不正な場合の動作は未定義です。
*/

void getColors(uint8_t value, uint8_t scheme, uint8_t *foreground, uint8_t *background)
{
	uint8_t original[] = {8, 255, 1, 255, 2, 255, 3, 255, 4, 255, 5, 255, 6, 255, 7, 255, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 255, 0, 255, 0};
	uint8_t blackwhite[] = {232, 255, 234, 255, 236, 255, 238, 255, 240, 255, 242, 255, 244, 255, 246, 0, 248, 0, 249, 0, 250, 0, 251, 0, 252, 0, 253, 0, 254, 0, 255, 0};
	uint8_t bluered[] = {235, 255, 63, 255, 57, 255, 93, 255, 129, 255, 165, 255, 201, 255, 200, 255, 199, 255, 198, 255, 197, 255, 196, 255, 196, 255, 196, 255, 196, 255, 196, 255};
	uint8_t *schemes[] = {original, blackwhite, bluered};
	// modify the 'pointed to' variables (using a * on the left hand of the assignment)
	*foreground = *(schemes[scheme] + (1 + value * 2) % sizeof(original));
	*background = *(schemes[scheme] + (0 + value * 2) % sizeof(original));
	// alternatively we could have returned a struct with two variables
}

/* @doc @function getDigitCount @level L4
 * @logical この関数は、与えられた整数の桁数を計算して返します。
 * @precise 入力された整数を10で割り続け、割り切れるまでの回数をカウントします。
 * @unambiguous 整数を10で割る操作を繰り返し、割り切れるまでの繰り返し回数を桁数として返します。
 * @exhaustive 関数は常に0以上の整数の桁数を返し、負の数や例外は考慮されていません。
*/

uint8_t getDigitCount(uint32_t number)
{
	uint8_t count = 0;
	do
	{
		number /= 10;
		count += 1;
	} while (number);
	return count;
}

/* @doc @function drawBoard @level L4
 * @logical この関数は、2048ゲームのボードを指定されたカラースキームで描画し、スコアを表示します。
 * @precise ボードの各セルに対して、getColors関数を使用して前景色と背景色を決定し、ANSIエスケープコードで色を設定して数値を表示します。
 * @unambiguous ボードの各セルに対して、数値が0でない場合は2のべき乗を表示し、0の場合はドットを表示します。
 * @exhaustive 関数はボードの状態を表示し、色の設定をリセットしますが、エラーや例外の処理は行いません。
*/

void drawBoard(uint8_t board[SIZE][SIZE], uint8_t scheme, uint32_t score)
{
	uint8_t x, y, fg, bg;
	printf("\033[H"); // move cursor to 0,0
	printf("2048.c %17u pts\n\n", score);
	for (y = 0; y < SIZE; y++)
	{
		for (x = 0; x < SIZE; x++)
		{
			// send the addresses of the foreground and background variables,
			// so that they can be modified by the getColors function
			getColors(board[x][y], scheme, &fg, &bg);
			printf("\033[38;5;%u;48;5;%um", fg, bg); // set color
			printf("       ");
			printf("\033[m"); // reset all modes
		}
		printf("\n");
		for (x = 0; x < SIZE; x++)
		{
			getColors(board[x][y], scheme, &fg, &bg);
			printf("\033[38;5;%u;48;5;%um", fg, bg); // set color
			if (board[x][y] != 0)
			{
				uint32_t number = 1 << board[x][y];
				uint8_t t = 7 - getDigitCount(number);
				printf("%*s%u%*s", t - t / 2, "", number, t / 2, "");
			}
			else
			{
				printf("   ·   ");
			}
			printf("\033[m"); // reset all modes
		}
		printf("\n");
		for (x = 0; x < SIZE; x++)
		{
			getColors(board[x][y], scheme, &fg, &bg);
			printf("\033[38;5;%u;48;5;%um", fg, bg); // set color
			printf("       ");
			printf("\033[m"); // reset all modes
		}
		printf("\n");
	}
	printf("\n");
	printf("        ←,↑,→,↓ or q        \n");
	printf("\033[A"); // one line up
	printf("fg=%u bg=%u\n", fg, bg);
}

/* @doc @function findTarget @level L4
 * @logical この関数は、指定された配列内で特定の条件に基づいてターゲット位置を見つける。
 * @precise 配列内の指定位置から逆方向に探索し、非ゼロかつ異なる値を持つ最初の要素の位置を返す。
 * @unambiguous 指定された位置から逆方向に進み、非ゼロで現在の位置と異なる値を持つ最初の要素を見つけた場合、その位置を返す。
 * @exhaustive 位置が0の場合はそのまま返し、探索中に停止位置に達した場合はその位置を返し、条件に合う要素が見つからない場合は元の位置を返す。
*/

uint8_t findTarget(uint8_t array[SIZE], uint8_t x, uint8_t stop)
{
	uint8_t t;
	// if the position is already on the first, don't evaluate
	if (x == 0)
	{
		return x;
	}
	for (t = x - 1;; t--)
	{
		if (array[t] != 0)
		{
			if (array[t] != array[x])
			{
				// merge is not possible, take next position
				return t + 1;
			}
			return t;
		}
		else
		{
			// we should not slide further, return this one
			if (t == stop)
			{
				return t;
			}
		}
	}
	// we did not find a target
	return x;
}

/* @doc @function slideArray @level L4
 * @logical この関数は、指定された配列内の要素をスライドさせ、可能であれば隣接する同じ値をマージします。
 * @precise 配列内の各要素を順にチェックし、ゼロでない場合はターゲット位置を見つけ、必要に応じて移動またはマージを行います。
 * @unambiguous 配列の各要素を左から右にチェックし、ゼロでない要素を可能な限り左に移動し、同じ値が隣接する場合はマージしてスコアを更新します。
 * @exhaustive 関数は、要素を移動またはマージした場合にtrueを返し、何も変更がない場合はfalseを返します。
*/

bool slideArray(uint8_t array[SIZE], uint32_t *score)
{
	bool success = false;
	uint8_t x, t, stop = 0;

	for (x = 0; x < SIZE; x++)
	{
		if (array[x] != 0)
		{
			t = findTarget(array, x, stop);
			// if target is not original position, then move or merge
			if (t != x)
			{
				// if target is zero, this is a move
				if (array[t] == 0)
				{
					array[t] = array[x];
				}
				else if (array[t] == array[x])
				{
					// merge (increase power of two)
					array[t]++;
					// increase score
					*score += 1 << array[t];
					// set stop to avoid double merge
					stop = t + 1;
				}
				array[x] = 0;
				success = true;
			}
		}
	}
	return success;
}

/* @doc @function rotateBoard @level L4
 * @logical この関数は、2次元配列で表現されたボードを90度時計回りに回転させます。
 * @precise ボードの外周から内側に向かって、各要素を一時変数を用いて4つの位置を交換することで回転を実現します。
 * @unambiguous ボードの各層に対して、4つの位置を順次交換することで90度回転を行います。
 * @exhaustive 関数はボードを直接変更し、戻り値はありません。
*/

void rotateBoard(uint8_t board[SIZE][SIZE])
{
	uint8_t i, j, n = SIZE;
	uint8_t tmp;
	for (i = 0; i < n / 2; i++)
	{
		for (j = i; j < n - i - 1; j++)
		{
			tmp = board[i][j];
			board[i][j] = board[j][n - i - 1];
			board[j][n - i - 1] = board[n - i - 1][n - j - 1];
			board[n - i - 1][n - j - 1] = board[n - j - 1][i];
			board[n - j - 1][i] = tmp;
		}
	}
}

/* @doc @function moveUp @level L4
 * @logical この関数は、ゲームボードの各列を上方向にスライドさせる機能を提供します。
 * @precise 関数は、ボードの各列に対して `slideArray` 関数を呼び出し、スライドが成功したかどうかを示すフラグを更新します。
 * @unambiguous 関数は、ボードの各列を順に処理し、スライドが成功した場合に `true` を返します。
 * @exhaustive 関数は、少なくとも1つの列がスライドに成功した場合に `true` を返し、成功しなかった場合は `false` を返します。
*/

bool moveUp(uint8_t board[SIZE][SIZE], uint32_t *score)
{
	bool success = false;
	uint8_t x;
	for (x = 0; x < SIZE; x++)
	{
		success |= slideArray(board[x], score);
	}
	return success;
}

/* @doc @function moveLeft @level L4
 * @logical この関数は、ゲームボード上のタイルを左方向に移動させるための処理を行います。
 * @precise ボードを90度回転させてから上方向にタイルを移動し、その後ボードを元の向きに戻すために3回回転させます。
 * @unambiguous ボードを左に移動するために、まず90度回転して上方向に移動し、3回回転して元の向きに戻します。
 * @exhaustive 関数はタイルの移動が成功したかどうかを示すブール値を返しますが、エラー処理は実装されていません。
*/

bool moveLeft(uint8_t board[SIZE][SIZE], uint32_t *score)
{
	bool success;
	rotateBoard(board);
	success = moveUp(board, score);
	rotateBoard(board);
	rotateBoard(board);
	rotateBoard(board);
	return success;
}

/* @doc @function moveDown @level L4
 * @logical この関数は、ゲームボードを下方向に移動させるための処理を行います。
 * @precise ボードを180度回転させてから上方向に移動し、その後再び180度回転させて元の向きに戻します。
 * @unambiguous ボードを2回回転させた後、moveUp関数を呼び出して移動を行い、再度2回回転させて元の状態に戻します。
 * @exhaustive 成功した場合はtrueを返し、失敗した場合はfalseを返します。スコアはmoveUp関数の処理によって更新されます。
*/

bool moveDown(uint8_t board[SIZE][SIZE], uint32_t *score)
{
	bool success;
	rotateBoard(board);
	rotateBoard(board);
	success = moveUp(board, score);
	rotateBoard(board);
	rotateBoard(board);
	return success;
}

/* @doc @function moveRight @level L4
 * @logical この関数はゲームボードを右に移動させるための処理を行います。
 * @precise ボードを3回回転させた後、上方向への移動処理を行い、最後にもう一度ボードを回転させます。
 * @unambiguous ボードを3回回転させてから上方向に移動し、再度1回回転させることで右方向への移動を実現します。
 * @exhaustive 関数は右方向への移動が成功したかどうかを示すブール値を返します。
*/

bool moveRight(uint8_t board[SIZE][SIZE], uint32_t *score)
{
	bool success;
	rotateBoard(board);
	rotateBoard(board);
	rotateBoard(board);
	success = moveUp(board, score);
	rotateBoard(board);
	return success;
}

/* @doc @function findPairDown @level L4
 * @logical この関数は、2次元配列内で縦に隣接する同じ値のペアを探します。
 * @precise ボード上の各列を上から下に走査し、縦に隣接する要素が等しいかを確認します。
 * @unambiguous 同じ列内で上下に隣接する2つの要素が等しい場合にtrueを返します。
 * @exhaustive 同じ値のペアが見つかった場合はtrueを返し、見つからない場合はfalseを返します。
*/

bool findPairDown(uint8_t board[SIZE][SIZE])
{
	bool success = false;
	uint8_t x, y;
	for (x = 0; x < SIZE; x++)
	{
		for (y = 0; y < SIZE - 1; y++)
		{
			if (board[x][y] == board[x][y + 1])
				return true;
		}
	}
	return success;
}

/* @doc @function countEmpty @level L4
 * @logical この関数は、与えられた2次元配列内の空の要素（値が0の要素）の数をカウントします。
 * @precise 2重のforループを用いて配列の各要素を走査し、要素が0であればカウントをインクリメントします。
 * @unambiguous 配列内の各要素をチェックし、0である場合に限りカウントを増やします。
 * @exhaustive 関数は、配列内の0の要素の数を返し、エラーや例外処理は行いません。
*/

uint8_t countEmpty(uint8_t board[SIZE][SIZE])
{
	uint8_t x, y;
	uint8_t count = 0;
	for (x = 0; x < SIZE; x++)
	{
		for (y = 0; y < SIZE; y++)
		{
			if (board[x][y] == 0)
			{
				count++;
			}
		}
	}
	return count;
}

/* @doc @function gameEnded @level L4
 * @logical この関数は、ゲームが終了したかどうかを判定します。
 * @precise ボード上に空きマスがあるか、隣接するペアが存在するかをチェックし、ゲームが続行可能かを判断します。
 * @unambiguous ボードに空きがないかつ隣接するペアがない場合にゲーム終了と判定します。
 * @exhaustive 空きマスがある場合や隣接するペアがある場合はfalseを返し、どちらもない場合はtrueを返します。
*/

bool gameEnded(uint8_t board[SIZE][SIZE])
{
	bool ended = true;
	if (countEmpty(board) > 0)
		return false;
	if (findPairDown(board))
		return false;
	rotateBoard(board);
	if (findPairDown(board))
		ended = false;
	rotateBoard(board);
	rotateBoard(board);
	rotateBoard(board);
	return ended;
}

/* @doc @function addRandom @level L4
 * @logical この関数は、与えられたボード上のランダムな空きセルに1または2を追加します。
 * @precise ボード上の空きセルをリスト化し、その中からランダムに選んだセルに1または2を配置します。
 * @unambiguous ボードの空きセルを見つけ、それらの中からランダムに選んだセルに1または2を配置します。
 * @exhaustive 空きセルが存在する場合、ランダムに選んだセルに1または2を配置し、空きセルがない場合は何も行いません。
*/

void addRandom(uint8_t board[SIZE][SIZE])
{
	static bool initialized = false;
	uint8_t x, y;
	uint8_t r, len = 0;
	uint8_t n, list[SIZE * SIZE][2];

	if (!initialized)
	{
		srand(time(NULL));
		initialized = true;
	}

	for (x = 0; x < SIZE; x++)
	{
		for (y = 0; y < SIZE; y++)
		{
			if (board[x][y] == 0)
			{
				list[len][0] = x;
				list[len][1] = y;
				len++;
			}
		}
	}

	if (len > 0)
	{
		r = rand() % len;
		x = list[r][0];
		y = list[r][1];
		n = (rand() % 10) / 9 + 1;
		board[x][y] = n;
	}
}

/* @doc @function initBoard @level L4
 * @logical この関数は、ゲームボードを初期化し、ランダムな位置に2つの値を追加します。
 * @precise 2次元配列boardのすべての要素を0に設定し、addRandom関数を2回呼び出してランダムな位置に値を追加します。
 * @unambiguous 関数は、指定されたサイズのボードを0で初期化し、addRandom関数を用いて2つのランダムな位置に値を配置します。
 * @exhaustive 関数は戻り値を持たず、board配列の状態を変更します。エラー処理は実装されていません。
*/

void initBoard(uint8_t board[SIZE][SIZE])
{
	uint8_t x, y;
	for (x = 0; x < SIZE; x++)
	{
		for (y = 0; y < SIZE; y++)
		{
			board[x][y] = 0;
		}
	}
	addRandom(board);
	addRandom(board);
}

/* @doc @function setBufferedInput @level L4
 * @logical この関数は、標準入力のバッファリングを有効または無効にする。
 * @precise 引数がtrueの場合は以前の設定を復元し、falseの場合は標準入力のカノニカルモードとエコーを無効にする。
 * @unambiguous 引数に基づいて標準入力の設定を変更し、バッファリングの状態を切り替える。
 * @exhaustive 引数がtrueでバッファリングが無効な場合は設定を復元し、falseで有効な場合は設定を変更する。
*/

void setBufferedInput(bool enable)
{
	static bool enabled = true;
	static struct termios old;
	struct termios new;

	if (enable && !enabled)
	{
		// restore the former settings
		tcsetattr(STDIN_FILENO, TCSANOW, &old);
		// set the new state
		enabled = true;
	}
	else if (!enable && enabled)
	{
		// get the terminal settings for standard input
		tcgetattr(STDIN_FILENO, &new);
		// we want to keep the old setting to restore them at the end
		old = new;
		// disable canonical mode (buffered i/o) and local echo
		new.c_lflag &= (~ICANON & ~ECHO);
		// set the new settings immediately
		tcsetattr(STDIN_FILENO, TCSANOW, &new);
		// set the new state
		enabled = false;
	}
}

/* @doc @function testSucceed @level L4
 * @logical この関数は、事前に定義されたテストケースを使用して、`slideArray`関数の出力が期待通りであるかを検証します。
 * @precise 関数は、データセットから入力と期待される出力を取得し、`slideArray`関数を呼び出して結果を比較します。すべてのテストケースが成功した場合にtrueを返します。
 * @unambiguous 関数は、各テストケースで`slideArray`の出力と期待される出力を比較し、一致しない場合はfalseを返します。
 * @exhaustive すべてのテストケースが成功した場合はtrueを返し、失敗した場合はfalseを返します。失敗した場合には、期待される出力と実際の出力を表示します。
*/

bool testSucceed()
{
	uint8_t array[SIZE];
	// these are exponents with base 2 (1=2 2=4 3=8)
	// data holds per line: 4x IN, 4x OUT, 1x POINTS
	uint8_t data[] = {
		0, 0, 0, 1, 1, 0, 0, 0, 0,
		0, 0, 1, 1, 2, 0, 0, 0, 4,
		0, 1, 0, 1, 2, 0, 0, 0, 4,
		1, 0, 0, 1, 2, 0, 0, 0, 4,
		1, 0, 1, 0, 2, 0, 0, 0, 4,
		1, 1, 1, 0, 2, 1, 0, 0, 4,
		1, 0, 1, 1, 2, 1, 0, 0, 4,
		1, 1, 0, 1, 2, 1, 0, 0, 4,
		1, 1, 1, 1, 2, 2, 0, 0, 8,
		2, 2, 1, 1, 3, 2, 0, 0, 12,
		1, 1, 2, 2, 2, 3, 0, 0, 12,
		3, 0, 1, 1, 3, 2, 0, 0, 4,
		2, 0, 1, 1, 2, 2, 0, 0, 4};
	uint8_t *in, *out, *points;
	uint8_t t, tests;
	uint8_t i;
	bool success = true;
	uint32_t score;

	tests = (sizeof(data) / sizeof(data[0])) / (2 * SIZE + 1);
	for (t = 0; t < tests; t++)
	{
		in = data + t * (2 * SIZE + 1);
		out = in + SIZE;
		points = in + 2 * SIZE;
		for (i = 0; i < SIZE; i++)
		{
			array[i] = in[i];
		}
		score = 0;
		slideArray(array, &score);
		for (i = 0; i < SIZE; i++)
		{
			if (array[i] != out[i])
			{
				success = false;
			}
		}
		if (score != *points)
		{
			success = false;
		}
		if (success == false)
		{
			for (i = 0; i < SIZE; i++)
			{
				printf("%u ", in[i]);
			}
			printf("=> ");
			for (i = 0; i < SIZE; i++)
			{
				printf("%u ", array[i]);
			}
			printf("(%u points) expected ", score);
			for (i = 0; i < SIZE; i++)
			{
				printf("%u ", in[i]);
			}
			printf("=> ");
			for (i = 0; i < SIZE; i++)
			{
				printf("%u ", out[i]);
			}
			printf("(%u points)\n", *points);
			break;
		}
	}
	if (success)
	{
		printf("All %u tests executed successfully\n", tests);
	}
	return success;
}

/* @doc @function signal_callback_handler @level L4
 * @logical この関数は、シグナルを受け取った際にプログラムを終了するための処理を行います。
 * @precise シグナルを受け取ると、終了メッセージを表示し、バッファード入力を有効にし、カーソルを表示してからプログラムを終了します。
 * @unambiguous シグナル番号を受け取り、終了メッセージを表示し、入力モードをリセットしてから指定されたシグナル番号でプログラムを終了します。
 * @exhaustive この関数は、シグナルを受け取ると必ず終了メッセージを表示し、入力モードをリセットし、プログラムを終了します。
*/

void signal_callback_handler(int signum)
{
	printf("         TERMINATED         \n");
	setBufferedInput(true);
	// make cursor visible, reset all modes
	printf("\033[?25h\033[m");
	exit(signum);
}

/* @doc @function main @level L4
 * @logical この関数は、コンソール上で2048ゲームを実行し、ユーザーの入力に応じてゲームの状態を更新します。
 * @precise コマンドライン引数を解析してゲームのモードを設定し、ゲームループ内でユーザーのキーボード入力に基づいてボードの状態を更新し、ゲームが終了するまでループを続けます。
 * @unambiguous ユーザーが指定したオプションに基づいてゲームの色スキームを設定し、ユーザーの入力に応じてボードを移動させ、ゲームの終了条件を満たすまで繰り返します。
 * @exhaustive コマンドライン引数が不正な場合はエラーメッセージを表示して終了し、ゲーム終了時には終了メッセージを表示します。正常に終了した場合はEXIT_SUCCESSを返します。
*/

int main(int argc, char *argv[])
{
	uint8_t board[SIZE][SIZE];
	uint8_t scheme = 0;
	uint32_t score = 0;
	int c;
	bool success;

	// handle the command line options
	if (argc > 1)
	{
		if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
		{
			printf("Usage: 2048 [OPTION] | [MODE]\n");
			printf("Play the game 2048 in the console\n\n");
			printf("Options:\n");
			printf("  -h,  --help       Show this help message.\n");
			printf("  -v,  --version    Show version number.\n\n");
			printf("Modes:\n");
			printf("  bluered      Use a blue-to-red color scheme (requires 256-color terminal support).\n");
			printf("  blackwhite   The black-to-white color scheme (requires 256-color terminal support).\n");
			return EXIT_SUCCESS;
		}
		else if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0)
		{
			printf("2048.c version %s\n", VERSION);
			return EXIT_SUCCESS;
		}
		else if (strcmp(argv[1], "blackwhite") == 0)
		{
			scheme = 1;
		}
		else if (strcmp(argv[1], "bluered") == 0)
		{
			scheme = 2;
		}
		else if (strcmp(argv[1], "test") == 0)
		{
			return testSucceed() ? EXIT_SUCCESS : EXIT_FAILURE;
		}
		else
		{
			printf("Invalid option: %s\n\nTry '%s --help' for more options.\n", argv[1], argv[0]);
			return EXIT_FAILURE;
		}
	}

	// make cursor invisible, erase entire screen
	printf("\033[?25l\033[2J");

	// register signal handler for when ctrl-c is pressed
	signal(SIGINT, signal_callback_handler);

	initBoard(board);
	setBufferedInput(false);
	drawBoard(board, scheme, score);
	while (true)
	{
		c = getchar();
		if (c == EOF)
		{
			puts("\nError! Cannot read keyboard input!");
			break;
		}
		switch (c)
		{
		case 97:  // 'a' key
		case 104: // 'h' key
		case 68:  // left arrow
			success = moveLeft(board, &score);
			break;
		case 100: // 'd' key
		case 108: // 'l' key
		case 67:  // right arrow
			success = moveRight(board, &score);
			break;
		case 119: // 'w' key
		case 107: // 'k' key
		case 65:  // up arrow
			success = moveUp(board, &score);
			break;
		case 115: // 's' key
		case 106: // 'j' key
		case 66:  // down arrow
			success = moveDown(board, &score);
			break;
		default:
			success = false;
		}
		if (success)
		{
			drawBoard(board, scheme, score);
			usleep(150 * 1000); // 150 ms
			addRandom(board);
			drawBoard(board, scheme, score);
			if (gameEnded(board))
			{
				printf("         GAME OVER          \n");
				break;
			}
		}
		if (c == 'q')
		{
			printf("        QUIT? (y/n)         \n");
			c = getchar();
			if (c == 'y')
			{
				break;
			}
			drawBoard(board, scheme, score);
		}
		if (c == 'r')
		{
			printf("       RESTART? (y/n)       \n");
			c = getchar();
			if (c == 'y')
			{
				initBoard(board);
				score = 0;
			}
			drawBoard(board, scheme, score);
		}
	}
	setBufferedInput(true);

	// make cursor visible, reset all modes
	printf("\033[?25h\033[m");

	return EXIT_SUCCESS;
}
