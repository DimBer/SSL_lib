 

1) COMPILE SOURCE CODE INTO OBJECT FILES

$ gcc -c -Wall -Werror -fPIC Tuned_RwR.c csr_handling.c G_slice_extraction.c my_IO.c parameter_opt.c -lblas -lm 


2) LOAD OBJECT FILES INTO PROPERLY NAMED DYNAMICALLY LINKED LIBRARY

$ gcc -shared -o librwr.so Tuned_RwR.o my_IO.o csr_handling.o G_slice_extraction.o parameter_opt.o

3) ALSO PUT HEADER FILE TunedRwR.h TOGETHER WITH LIBRARY SINCE IT CONTAINS THE DECLARATION

4) COMPILE MAIN WHILE PROVIDING PATHS FOR .so LIBRARY (-L) AND #include (-I)

$ gcc -I/home/dimitris/Desktop/my_libs  test.c -o test -L/home/dimitris/Desktop/my_libs -lrwr -lblas

5) BEFORE EXECUTION YOU NEED TO LET THE PROGRAM KNOW WHERE THE LIBRARY IS

$ LD_LIBRARY_PATH=/home/...
$ export LD_LIBRARY_PATH 
$ ./test ...



