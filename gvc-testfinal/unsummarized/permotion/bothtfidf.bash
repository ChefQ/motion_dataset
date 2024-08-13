#!/bin/bash


for k in both ; do
  for f in tfidf ; do
    python3 predict-motion-both.py >& log.$k.$f
  done
done
