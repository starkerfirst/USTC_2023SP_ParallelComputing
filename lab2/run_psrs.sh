mpicc psrs.c -g -o psrs
mpirun --allow-run-as-root -np 5 -v psrs