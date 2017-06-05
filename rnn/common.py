# coding: utf-8

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def print_bold(str):
	print(stdout.BOLD + str + stdout.END)

bucket_sizes = [10, 20, 40, 100, 200]
ID_PAD = -1
ID_BOS = 0
ID_EOS = 1