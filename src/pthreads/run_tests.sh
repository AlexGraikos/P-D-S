# A simple bash script that for running experiments
# Note: To run the script make sure it you have execution rights 
# (use: chmod u+x run_tests.sh to give execution rights) 
!/bin/bash

NAME=$(hostname)
DATE=$(date "+%Y-%m-%d-%H:%M:%S")
FILE_PREF=$NAME-$DATE-test-tree

echo $NAME
echo $DATE

make clean; make
# run cube experiments
for N in 1000000 2000000 ; do \
    for P in 68 128 ; do \
        for L in 10 20 ; do \
            echo cube N=$N && ./test_octree $N 0 $P 3 $L >> $FILE_PREF-cube.txt ; \
	done ; \
    done ; \
done ;
# run octant experiments
for N in 1000000 2000000 ; do \
    for P in 68 128 ; do \
        for L in 10 20 ; do \
           echo Plummer N=$N && ./test_octree $N 1 $P 3 $L >> $FILE_PREF-plummer.txt ; \
        done ; \
    done ; \
done ; 
