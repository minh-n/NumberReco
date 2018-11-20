import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time 
import argparse
import sys

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="automatic picture analysis")
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
	group.add_argument("-q", "--quiet", action="store_true", help="reduce output verbosity")
	parser.add_argument("-d", "--data", type=str, help="the .mat file to analyze")
	args = parser.parse_args()

	if not args.data:
		raise IOError("Please specify a data.mat with the -d or --data option")
	
	try:
		with loadmat(args.data) as data:
			print("TODO")
	except (IOError, OSError) as error:
		sys.stderr.write(error.strerror + "\n")
