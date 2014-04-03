

all: datadep.svg

.PHONY: clean

clean:
	rm datadep.svg

datadep.svg: datadep.dot
		dot -Tsvg -odatadep.svg datadep.dot
