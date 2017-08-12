import sys

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def printb(string):
	print(stdout.BOLD + string + stdout.END)

def printr(string):
	sys.stdout.write("\r" + stdout.CLEAR)
	sys.stdout.write(string)
	sys.stdout.flush()

bucket_sizes = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 110), (200, 210)]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3