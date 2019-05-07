#!/bin/bash

opt=$1

if [ -z "$opt" ]
then
    echo "opt is required"
else
    for i in "$@"
    do
        if [[ $i -eq 0 ]]
        then
            continue
        fi

        if test $opt == "connect"
        then
            echo "ssh -N -f -L localhost:$i:localhost:$i leang@barbulle"
            ssh -N -f -L localhost:$i:localhost:$i leang@barbulle

        elif test $opt == "disconnect"
        then
            echo "lsof -n -i4TCP:$i | grep LISTEN | awk '{ print $2 }' | xargs kill"
            lsof -n -i4TCP:$i | grep LISTEN | awk '{ print $2 }' | xargs kill
        fi
    done
fi