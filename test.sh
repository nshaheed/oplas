SLURM_ARRAY_TASK_ID=64

TEST=$(echo "scale=10; 1/${SLURM_ARRAY_TASK_ID}" | bc)
result=$(echo "scale=10; 1/64" | bc)

LOAD_FRAC=$(echo "scale=10; 1/${SLURM_ARRAY_TASK_ID}" | bc)


echo ${TEST}
echo ${result}
echo ${LOAD_FRAC}
echo "DONE"
