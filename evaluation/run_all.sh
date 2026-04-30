#!/bin/bash

models=(
    # "/datas/models/Pangea-7B/"
)

tasks=(
    # "cvqa"   "cvqa_translated"
)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        bash script.sh "${model}" "${task}"
    done
done