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

bucket_sizes = [10, 20, 40, 100, 200]
ID_PAD = -1
ID_BOS = 0
ID_EOS = 1