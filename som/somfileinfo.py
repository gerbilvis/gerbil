# Simple utility to read binary SOM file header from gerbil SOM files.
#
# Author: Georg Altmann <georg.altmann@fau.de>


import math
import sys
import struct
from enum import Enum

som_type_table = [ 
	"BAD SOM TYPE",
	"SOM_SQUARE",
	"SOM_CUBE",
	"SOM_TESSERACT" 
	]

data_type_table = [ "bad data type", "32-bit IEEE 574 float" ]

class ExitStatus:
	Success   = 0
	BadUsage  = 1
	BadHeader = 2

def lookup(table, index):
	if index >= len(table):
		index = 0
	return table[index]

def main():
	if len(sys.argv) != 2:
		print_usage()
		sys.exit(ExitStatus.BadUsage)

	som_file_arg = sys.argv[1]
	buf = b''
	with open(som_file_arg, 'rb') as f:
		buf = f.read(32)

	if len(buf) != 32:
		print("error: no SOM file header, file too small", file=sys.stderr)
		sys.exit(ExitStatus.BadHeader)

	# read magic and version only
	magic_str, version = \
			struct.unpack('<16si', buf[0:20])

	magic_str = magic_str.decode("ascii")
	magic_good_str = "bad"
	if magic_str == "gerbilsom       ":
		magic_good_str = "good"
	else:
		print("error: bad SOM file magic string in header", file=sys.stderr)
		sys.exit(ExitStatus.BadHeader)

	if version != 3:
		print("error: SOM file version %d not supported, "
				"expected version 3" % version, file=sys.stderr)
		sys.exit(ExitStatus.BadHeader)

	# read the rest of the header, now that version and size are good
	_, _, data_type, som_type, size = \
			struct.unpack('<16siiii', buf)

	print("magic str (%s): '%s'" % (magic_good_str, magic_str))
	print("version:         %d"  % version)
	print("data type:       %s (%d)"  % 
			(lookup(data_type_table, data_type), data_type))
	print("som type:        %s (%d)"  % 
			(lookup(som_type_table, som_type), som_type))
	print("size (#neurons): %d"  % size)
	config_str = "unknown (cannot handle configuration for this SOM type)"
	if som_type == 1: # SQUARE
		dsize = int(math.sqrt(size))
		if dsize * dsize == size:
			config_str = "%d x %d neurons" % (dsize, dsize)
	elif som_type == 2: # CUBE
		dsize = int(math.pow(size, 1.0/3.0)+0.5)
		if dsize * dsize * dsize == size:
			config_str = "%d x %d x %d neurons" % (dsize, dsize, dsize)
	# TODO: tesseract SOM configuration
	print("configuration:   %s" % config_str)

	# As an additional feature, we might read the rest of the file, check for
	# correct length and valid float values in the future here.

def print_usage():
	print("usage: somfileinfo.py somfile", file=sys.stderr)


if __name__ == "__main__":
	main()
