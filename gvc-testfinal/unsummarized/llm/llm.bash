#!/bin/bash


for k in support opposition ; do
    sed -i -e 's/^key = .*/key = "'$k'"/' llmclassificationtutorial.py
    python3 llmclassificationtutorial.py
done
