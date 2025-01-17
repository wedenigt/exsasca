#!/bin/bash

# we initialize with a left-linear vtree, this works well for mixcolumns
# for other functions, we might want to try other vtrees

vtree=left
../bin/sdd-linux -c ./out.cnf -R $vtree.sdd -W $vtree.vtree -p -t $vtree