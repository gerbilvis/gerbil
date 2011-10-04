#!/usr/local/bin/gnuplot -persist
plot "/home/knorx/UNI/Studienarbeit/working_directory/code/reflectance/ms_edges/GTM/data/Modes.dat" using 1:2:3 with points pt 6 lc variable title 'Modes', \
 "/home/knorx/UNI/Studienarbeit/working_directory/code/reflectance/ms_edges/GTM/data/Means.dat" using 1:2:3 with points lc variable title  'Means'
#EOF