# Makefile

IDIR = include
CC = gcc
CFLAGS = -I$(IDIR) -lblas -lpthread -lm -Wall -g

SDIR = src
ODIR = obj

_SOURCES = my_utils.c AdaDIF.c SSL.c my_IO.c Tuned_RwR.c csr_handling.c comp_engine.c parameter_opt.c  
SOURCES = $(patsubst %,$(SDIR)/%,$(_SOURCES))

_OBJECTS = $(_SOURCES:.c=.o)
OBJECTS = $(patsubst %,$(ODIR)/%,$(_OBJECTS))

_DEPS = my_utils.h AdaDIF.h Tuned_RwR.h my_IO.h csr_handling.h comp_engine.h parameter_opt.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

EXE = SSL

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXE): $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS) 

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o 





