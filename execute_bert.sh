#!/bin/bash
array_num_epochs=(3 4)
array_batch_size=(16 32)
array_learning_rate=(5e-5 3e-5 2e-5)

for epoch in ${array_num_epochs[@]}; do
    for bs in ${array_batch_size[@]}; do
        for lr in ${array_learning_rate[@]}; do
            python main.py with model_name="BERTLinearFreezeEmbeddings" num_epochs=$epoch batch_size=$bs learning_rate=$lr
        done
    done
done
