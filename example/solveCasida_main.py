#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Job script for solving Casida equation for BSE calculations.             #
#                                                                             #

import sys

# Please edit the src path for importing.
sys.path.append('/home/wenm/green-bse/src')

from bse import create_argument_parser, BSESolver, BSEConfig

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = BSEConfig.from_args(args)
    
    # Create and run solver
    solver = BSESolver(config)
    solver.run()


if __name__ == "__main__":
    main()
