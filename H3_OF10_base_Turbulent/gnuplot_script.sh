#!/bin/bash

gnuplot -persist > /dev/null 2>&1 << EOF
	set title "Forces vs. Time"
	set xlabel "Time / Iteration"
	set ylabel "Force (N)"
	
	plot	"forces.txt" using 1:2 title 'Vertical Force',\

EOF
