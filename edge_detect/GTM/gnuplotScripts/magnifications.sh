#!/bin/bash
gnuplot -persist << EOF 
set style data pm3d
set style function pm3d
set pm3d implicit at b

set palette  model RGB rgbformulae 21,22,23
set title "Latent space" font "Helvetica,20"
set xlabel "X" font "Helvetica,15"
set ylabel "Y" font "Helvetica,15"
set zlabel "Magnification"  font "Helvetica,15" rotate by 90 left




unset logscale x 
unset logscale y 
unset logscale z
unset key
 
splot [:] [:] '$1' using 1:2:3  with linespoints ls 17;
# ,\
# '$2' using 1:2:3 with points title  'Means'


EOF