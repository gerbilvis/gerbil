#!/bin/bash
gnuplot -persist << EOF 

set title "Error value" 
set xlabel "Iteration X" 
set ylabel "Error Y" 
set zlabel "" 
unset logscale x 
unset logscale y 
unset logscale z
 
plot [:] [:] '$1' using 1:2  with linespoints 



EOF