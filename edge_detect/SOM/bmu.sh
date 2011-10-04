#!/bin/bash
gnuplot -persist << EOF 
set style data pm3d
set style function pm3d
set pm3d implicit at b
set palette gray
set title "Winnings" 
set xlabel "X" 
set ylabel "Y" 
set yrange [] reverse
set zlabel "Winnings (log)" 
unset logscale x 
unset logscale y 
unset logscale z
 
splot [:] [:] '$1' using 1:2:3  with linespoints


EOF