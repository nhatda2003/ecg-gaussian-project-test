import argparse
import sys

from helper_code import *
from main_model import train_dx_model

# Parse arguments.
def get_parser():
    description = 'Train the Challenge model(s).'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-n','--model_scenario_name',type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

# Run the code.
def run(args):
    print("ASDASd")
    #NHATedit 
    train_dx_model(args.data_folder, args.model_folder, args.model_scenario_name, args.verbose) ### Teams: Implement this function!!!
  

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))