#!/bin/bash


for k in both support opposition ; do
  if [[ "$k" == "both" ]] ; then
    sed -i -e 's/both=False/both=True/' predict-motion.py
  else
    sed -i -e 's/both=True/both=False/' predict-motion.py
  fi
  for f in tfidf embedding ; do
    sed -i -e 's/^key = .*/key = "'$k'"/' -e 's/^feature =.*/feature = "'$f'"/' predict-motion.py
    python3 predict-motion.py >& log.$k.$f
  done
done
