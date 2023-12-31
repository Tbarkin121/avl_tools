#======================================================#
# Makefile options for Xplot11 library                 #
#   With GNU gfortran and gcc compilers                #
#   Set up or select a set of compile                  #
#   options for your system                            # 
#                                                      #
#   Set PLTLIB to name of library                      #
#   Set DP for real precision                          #
#======================================================#

# Set library name (either libPlt.a or variant with compiler and precision)
#PLTLIB = libPlt.a
#PLTLIB = libPlt_gfortran.a
#PLTLIB = libPlt_gfortranSP.a
#PLTLIB = libPlt_gfortranDP.a
#PLTLIB = libPlt_gSP.a
PLTLIB = libPlt_gDP.a


# Some fortrans need trailing underscores in C interface symbols (see Xwin.c)
# This should work for most of the "unix" fortran compilers
DEFINE = -DUNDERSCORE

FC = gfortran
#CC = gcc
CC = cc

# Depending on your system and libraries you might specify an architecture flag
# to gcc/gfortran to give a compatible binary 32 bit or 64 bit 
# use -m32 for 32 bit binary, -m64 for 64 bit binary
MARCH =
#MARCH = -m64

# Fortran double precision (real) flag
DP =
DP = -fdefault-real-8

FFLAGS  = -O2 $(MARCH) $(DP)
CFLAGS  = -O2 $(MARCH) $(DEFINE)
CFLAGS0 = -O0 $(MARCH) $(DEFINE)

AR = ar r
RANLIB = ranlib 
LINKLIB = -L/usr/X11R6/lib -lX11 
WOBJ = Xwin2.o
WSRC = xwin11
