#!/bin/bash

repeat=10

while [ $repeat -gt 0 ]
do
	echo repeat $repeat
	repeat=$(($repeat - 1))
	python tf_autotrain_cnn.py 
done
