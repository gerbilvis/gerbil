#!/bin/bash
gnuplot -persist << EOF 
set style data pm3d
set style function pm3d
set pm3d implicit at b
set palette  model RGB rgbformulae 21,22,23
set title "Magnification Factors" 
set xlabel "Latent space X" 
set ylabel "Latent space Y" 
set zlabel "" 
unset logscale x 
unset logscale y 
unset logscale z
 

#1: magnifications
#2: meansModes
#3: modes
#4: means

splot [:] [:] '$1' using 1:2:3  with linespoints ,\
 '$2' using 1:2:3 with lines lt 2 title 'Means + Modes',\
 '$3' using 1:2:3 with points pt 6 lc 3 title 'Modes', \
 '$4' using 1:2:3 with points lc 4 title  'Means'

EOF