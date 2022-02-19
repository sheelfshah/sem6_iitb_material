DESTBIN = $(HOME)/bin/$(HOSTTYPE)
DESTINCLUDE = $(HOME)/include
DESTLIB = $(HOME)/lib/$(HOSTTYPE)
DESTMAN = $(HOME)/man


binaries = 

archives = $(trlibrary)

includes = truser.h

manpages = $(trmanpages)


trlibrary = libtr.a

trincludes = truser.h vec.h

trmanpages = tr.3


otherincludes = -I$(DESTINCLUDE)

INCLUDEDIRS = -I. $(otherincludes)

GCC = gcc

gccwarnings = -Wall -Wpointer-arith -Wconversion -Wstrict-prototypes -Wmissing-prototypes

GCCFLAGS = -O2 -g $(gccwarnings) $(INCLUDEDIRS)

CFLAGS = -g $(INCLUDEDIRS)


sources = \
  tr.c \
  vec.c

objects = \
  tr.o \
  vec.o


all: install


install: $(binaries) $(archives) $(includes) $(manpages)
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


$(trlibrary): $(objects)
	ar r $@ $(objects)
	-ranlib $@


dependencies: $(trincludes) $(sources)
	$(CC) -MM $(INCLUDEDIRS) $(sources) > $@


dist.tar: README dependencies Makefile $(sources) $(trincludes) $(trmanpages)
	tar cvf $@ README dependencies Makefile $(sources) $(trincludes) $(trmanpages)


makeobj: $(objects)


include dependencies
