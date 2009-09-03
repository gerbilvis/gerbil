CFLAGS+=`pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`
LDFLAGS+='-lboost_program_options'

OBJS=main.o multi_img.o mfams.o auxiliary.o io.o

PROG=gerbil

$(PROG): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(PROG)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

all: $(PROG)

default: $(PROG)

clean:
	rm -f $(OBJS) $(PROG)

