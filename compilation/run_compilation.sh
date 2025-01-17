#!/bin/bash

# this generates the CNF file (no byte indicator yet, just mixcolumns)
python ./create_mixcol_cnf.py 

# this takes the CNF file and generates the SDD out of it
./generate_base_sdd.sh

# this adds the byte indicators to the SDD. 
# we condition on g=254, but this is arbitrary. other values are possible
# this takes about 7h on our machine (see paper for more details)
python ./sdd_add_byte_indicators.py --g 254 --sdd-name left