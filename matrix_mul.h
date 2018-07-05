#ifndef _MATRIX_MUL_H
#define _MATRIX_MUL_H
  #define ASSERT
  #define BARRIERS
  //#define DEBUG
  //#define VERBOSE
  #define MASTER 0
  #define FROM_MASTER 1
  #define FROM_WORKER 2
  #define DEFAULT_SIZE 512

  #ifdef VERBOSE
    #ifndef DEBUG
      #define DEBUG
    #endif
    #define VER(...) do { fprintf(stderr, "VERBOSE: "); \
                          fprintf(stderr, __VA_ARGS__); \
                          fflush(stderr); } while(0)
    #define VER_matrix(...) do { fprintf(stderr, "VERBOSE: printing matrix\n"); \
                                    fprintf_matrix(stderr, __VA_ARGS__); \
                                    fflush(stderr); } while(0)
  #else
    #define VER(...) do {} while(0)
    #define VER_matrix(...) do {} while(0)
  #endif

  #ifdef DEBUG
    #define DBG(...) do { fprintf(stderr, "DEBUG: "); \
                          fprintf(stderr, __VA_ARGS__); \
                          fflush(stderr); } while(0)
    #define DBG_matrix(...) do { fprintf(stderr, "DEBUG: printing matrix\n"); \
                                    fprintf_matrix(stderr, __VA_ARGS__); \
                                    fflush(stderr); } while(0)
  #else
    #define DBG(...) do {} while(0)
    #define DBG_matrix(...) do {} while(0)
  #endif
#endif

long **init_long_matrix(int rows, int cols);
void fprintf_matrix(FILE *stream, long** matrix, int rows, int cols);
void usage(char* program_name);
int main(int argc, char *argv[]);
