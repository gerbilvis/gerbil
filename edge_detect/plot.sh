 #!/bin/bash
echo
echo "... _/\__/\__0>"
echo
echo "Bash version ${BASH_VERSION}"
MINPARAMS=1
MAXPARAMS=3

echo ">> Creating plot output directory [./plot/]"
echo ">> Remove this after first use"
mkdir plot

#check arguments
if [ $# -lt "$MINPARAMS" ]
then
  echo
  echo " ! This script needs at least $MINPARAMS command-line arguments!"
  echo " ! Usage: plot [single_plot | lower_bound upper_bund]"
  exit
fi 

if [ $# -gt "$MAXPARAMS" ]
then
  echo
  echo " ! Too much parameters! This script uses at max $MAXPARAMS command-line arguments!"
  echo " ! Usage: plot [single_plot | lower_bound upper_bund]"
  exit
fi 

#write single plot
if [ $# -eq 1 ]
then
  circo -Tps ./plot/iteration_$1.dot -o ./plot/iteration_$1.ps
  end=1
  echo -n " # Wrote 1 plot"
  echo
fi

#write $2 -$1 plots, from lower to upper range
if [ $# -eq 2 ]
then
  for (( c=$1; c<=$2;c++ ))
  do
  # circo -Tps iteration_$c.dot -o iteration_$c.ps
    neato -Tps ./plot/iteration_$c.dot -o ./plot/iteration_$c.ps
    echo -n " # Wrote  $c. plot"
    echo
  done
  end=$2
  let "end -=$1"
  let "end +=1"
fi

#write $2 -$1 plots, from lower to upper range using method in $3
if [ $# -eq 3 ]
then
  for (( c=$1; c<=$2;c++ ))
  do
  # circo -Tps iteration_$c.dot -o iteration_$c.ps
    $3 -Tps ./plot/iteration_$c.dot -o ./plot/iteration_$c.ps
    echo -n " # Wrote  $c. plot"
    echo
  done
  end=$2
  let "end -=$1"
  let "end +=1"
fi

echo " # Created $end .ps files"
echo
echo "<0_/\__/\_ ..."
echo
exit
