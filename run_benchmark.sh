#!/bin/bash
has_n_option=false
while getopts l:d:m:n: flag
do
    case "${flag}" in
        l) lang=${OPTARG};;
        d) dev=${OPTARG};;
        m) model=${OPTARG};;
        n) nmax=${OPTARG};has_n_option=true;;
        \?) echo "Unknown option -$OPTARG"; exit 1;;
    esac
done
if $has_n_option; then
    nmax=$nmax
else
    nmax=$((2**24))
fi
echo $lang
if [ $lang == "julia" ]; then
    echo "Benchmarking ${lang^} ${dev^^} accelerated ensemble ${model^^} solvers..."
    if [ $dev == "cpu" ];then
        cmd="./runner_scripts/${dev}/run_${model}_${lang}.sh ${nmax}"
        eval "$cmd"
    elif [ $model == "sde" ];then
        cmd="./runner_scripts/${dev}/run_${model}_${lang}.sh ${nmax}"
        eval "$cmd"
    else
        if [ -d "./data/${lang^}" ];
        then
            rm -f "./data/${lang^}"/*
            mkdir -p "./data/${lang^}"
        else
            mkdir -p "./data/${lang^}"
        fi
        cmd="./runner_scripts/${dev}/run_${model}_${lang}.sh ${nmax}"
        eval "$cmd"
    fi
elif [[ $lang == "jax"  ||  $lang == "pytorch" || $lang == "cpp" ]]; then
    if [[ $model != "ode" || $dev != "gpu" ]]; then
        echo "The benchmarking of ensemble ${model^^} solvers on ${dev^^} with ${lang} is not supported. Please use -m flag with \"ode\" and -d with \"gpu\"."
        exit 1
    else
        echo "Benchmarking ${lang^^} ${dev^^} accelerated ensemble ${model^^} solvers..."
        if [ -d "./data/${lang^^}/stiff" ] 
        then
            rm -rf "./data/${lang^^}"/stiff*
            mkdir -p "./data/${lang^^}/stiff"
        else
            mkdir -p "./data/${lang^^}/stiff"
        fi
        cmd="./runner_scripts/${dev}/run_${model}_${lang}.sh ${nmax}"
        eval "$cmd"
    fi
fi