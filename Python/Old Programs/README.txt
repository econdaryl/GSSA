This file contains GSSA code converted from the GSSA Matlab code. 

The Main_GSSA_1 file has been tested and appears to work for all implemented functions. Testing for Main_GSSA_N.py
was done using a vector of 1s for the random input (not a very good choice, but I think it did the job. In retrospect
using a saved text file of random values would have been better).

The Main_GSSA_N.py file has been tested for D values of 1 and 2. For D=1, everything appears to work. 
For D=2, the residual errors were slightly different from those obtained with the origional Matlab script. I am not sure if
this is due to numerical differences between the two softwares or a subltle error (perhaps a type def error?) somewhere in the
code. I have looked for a bug, but was unable to find one. Testing was done using the a20200.txt file as as the random input 
(note, this differs slightly from the Matlab file used in the origional code. The  a20200.txt file uses less percision. When
testing Main_GSSA_N.py, make sure you use a20200.txt for both the matlab and the python code for accurate comparison).

Four functions in the Num_Stab_Approx.py file were never implemented because I was unable to find and install a 
suibable linear program solver. I had looked at using lpsolve, a C based lp solver for python, but I had
difficulty getting it installed and eventually ran out of time.

Files:

GSSA.py Contains most of the functions needed for the GSSA algorithm
Main_GSSA_1.py and main_GSSA_N.py are the top level files, just like in the origional Matlab Code
Num_Stab_Approx.py contains the numerical solvers.

Running this code requires both the Numpy and Scipy libraries installed.

All coding was done using 64 bit Matlab version 2011 and 32 bit python 2.6.