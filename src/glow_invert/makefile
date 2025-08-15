#
FC = gfortran

 FFLAGS = -O3 -ffree-line-length-none -std=legacy
#FFLAGS = -g $(DBGFLAGS)

DBGFLAGS = -debug full -traceback
DBGFLAGS += -check bounds -check format -check output_conversion -check pointers -check uninit
DBGFLAGS += -fpe-all=0 # this traps all floating point exceptions

.SUFFIXES: .o .F .F90 .f .f90

%.o: %.f90
	$(FC) $(FFLAGS) -c  -o $@ $<
%.o: %.f
	$(FC) $(FFLAGS) -c  -o $@ $<
#
# Sources (in order of dependency):
#
SOURCES = GLOW/cglow.f90 GLOW/glow.f90 GLOW/bands.f90 GLOW/conduct.f90 GLOW/egrid.f90 GLOW/ephoto.f90 GLOW/etrans.f90 GLOW/exsect.f GLOW/fieldm.f GLOW/gchem.f90 GLOW/geomag.f90 GLOW/maxt.f90 GLOW/mzgrid.f90 GLOW/qback.f90 GLOW/rcolum.f90 GLOW/rout.f90 emout.f90 GLOW/snoem.f90 GLOW/snoemint.f90 GLOW/solzen.f90 GLOW/ssflux.f90 iri90_dlh.f GLOW/nrlmsise00.f

OBJS1 := $(addsuffix .o, $(basename $(SOURCES) glow_invert_tables.f90))
EXEC1 = glow_invert_tables.exe

tables: $(OBJS1)
	$(FC) -o $(EXEC1) $(OBJS1)


OBJS2 := $(addsuffix .o, $(basename $(SOURCES) glow_invert_airglow.f90))
EXEC2 = glow_invert_airglow.exe

airglow: $(OBJS2)
	$(FC) -o $(EXEC2) $(OBJS2)

clean:
	#rm -f *.o *.mod $(EXEC)
	rm GLOW/*.o *.o *.mod $(EXEC1) $(EXEC2)

