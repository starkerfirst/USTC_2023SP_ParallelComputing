mpicc fulladd.c -g -o fulladd
mpirun --allow-run-as-root -np 16 -v fulladd