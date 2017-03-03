# This script documents how to setup the dependencies for this
# package using software installed on CVMFS, e.g. on NERSC's PDSF system.

lsetup "lcgenv -p LCG_85b x86_64-slc6-gcc49-opt root_numpy" \
       "lcgenv -p LCG_85b x86_64-slc6-gcc49-opt decorator" \
       "lcgenv -p LCG_85b x86_64-slc6-gcc49-opt ipython" \
       "lcgenv -p LCG_85b x86_64-slc6-gcc49-opt jupyter" \
       "lcgenv -p LCG_85b x86_64-slc6-gcc49-opt scikitlearn" \
       "lcgenv -p LCG_85b x86_64-slc6-gcc49-opt matplotlib"

# Clean up all the duplicates
export PATH="$(perl -e 'print join(":", grep { not $seen{$_}++ } split(/:/, $ENV{PATH}))')"
export LD_LIBRARY_PATH="$(perl -e 'print join(":", grep { not $seen{$_}++ } split(/:/, $ENV{LD_LIBRARY_PATH}))')"
