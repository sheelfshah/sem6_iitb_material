MY_DIR = /home/sfshah/Desktop/sem6_iitb_material/ee709_vlsi_testing/bddlib
DESTBIN = $(MY_DIR)/bin
DESTINCLUDE = $(MY_DIR)/include
DESTLIB = $(MY_DIR)/lib/$(HOSTTYPE)
DESTMAN = $(MY_DIR)/man


binaries = 

archives = $(memlibrary)

includes = $(memincludes)

manpages = $(memmanpages)


memlibrary = libmem.a

memincludes = memuser.h

memmanpages = mem.3


otherincludes = -I$(MY_DIR)/include

INCLUDEDIRS = -I. $(otherincludes)

GCC = gcc

gccwarnings = -Wall -Wpointer-arith -Wconversion -Wstrict-prototypes -Wmissing-prototypes -Wtraditional

GCCFLAGS = -O2 -g $(gccwarnings) $(INCLUDEDIRS)

CFLAGS = -g $(INCLUDEDIRS)


sources = \
  memblock.c \
  memrec.c

objects = \
  memblock.o \
  memrec.o


all: install


install: $(archives) $(includes) $(manpages)
	for archive in $(archives); do \
	  mv $$archive $(DESTLIB); \
	done
	for include in $(includes); do \
	  cp $$include $(DESTINCLUDE); \
	done
	for manpage in $(manpages); do \
	  case $$manpage in \
	    *.0 ) cp $$manpage $(DESTMAN)/man0;; \
	    *.1 ) cp $$manpage $(DESTMAN)/man1;; \
	    *.2 ) cp $$manpage $(DESTMAN)/man2;; \
	    *.3 ) cp $$manpage $(DESTMAN)/man3;; \
	    *.4 ) cp $$manpage $(DESTMAN)/man4;; \
	    *.5 ) cp $$manpage $(DESTMAN)/man5;; \
	    *.6 ) cp $$manpage $(DESTMAN)/man6;; \
	    *.7 ) cp $$manpage $(DESTMAN)/man7;; \
	    *.8 ) cp $$manpage $(DESTMAN)/man8;; \
	  esac; \
	done


$(memlibrary): $(objects)
	ar r $@ $(objects)
	-ranlib $@


dependencies: $(bddincludes) $(sources)
	$(CC) -MM $(INCLUDEDIRS) $(sources) > $@


dist.tar: README dependencies Makefile $(sources) $(memincludes) memint.h $(memmanpages)
	tar cvf $@ README dependencies Makefile $(sources) $(memincludes) memint.h $(memmanpages)


makeobj: $(objects)


include dependencies
