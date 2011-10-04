#!/bin/bash

# example setup for SOM
../../core/vole edge_detect -I /home/knorx/UNI/Studienarbeit/CAVE_256/glass_tiles_ms/ --msiName glass_tiles_ms.txt --somWidth 10 --somHeight 12  --somRadiusStart 6 --somRadiusEnd 0.5 --somMaxIter 40000  --initialDegree 8 --graph_type="MESH"  --sw_model="PHI" --phi=0.0 -V 3  -A "SOM" -O /home/knorx/UNI/Studienarbeit/working_directory/gtm_test/ --withGraph false --withUMap false --scaleUDistance 1.0 --forceDD false --linearization "NONE"

# example setup for periodic Small World SOM with beta = 0.1 
# ../../core/vole edge_detect -I /home/knorx/UNI/Studienarbeit/CAVE_256/glass_tiles_ms/ --msiName glass_tiles_ms.txt --somWidth 10 --somHeight 12  --somRadiusStart 6 --somRadiusEnd 0.5 --somMaxIter 40000  --initialDegree 8 --graph_type="MESH_P"  --sw_model="BETA" --beta=0.0 -V 3  -A "SOM" -O /home/knorx/UNI/Studienarbeit/working_directory/gtm_test/ --withGraph false --withUMap false --scaleUDistance 1.0 --forceDD false --linearization "NONE"

# example setup for GTM
# ../../core/vole edge_detect -I /home/knorx/UNI/Studienarbeit/CAVE_256/egyptian_statue_ms/ --msiName egyptian_statue_ms.txt --rbfActFunc="GAUSSIAN" --rbfSize 4 --latentSize 15 --emIterations 30 --samplePercentage 0.4 --initialDegree 4  -V 3 --withUMap true --withGraph false -A "GTM" -O /home/knorx/UNI/Studienarbeit/working_directory/gtm_test/ --withGraph false --withUMap false --scaleUDistance 1.0 --forceDD false --linearization "NONE"

