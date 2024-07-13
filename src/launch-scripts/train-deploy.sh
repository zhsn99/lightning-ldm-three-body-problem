#!/bin/bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=128gb
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --output=slurm-%j-%N.out

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

source ~/.bashrc
conda activate lightning-ldm

MAIN_HOST=`hostname -s`
export MASTER_ADDR=$MAIN_HOST
export MASTER_PORT=`python free-port.py $MASTER_ADDR`

export NCCL_IB_DISABLE=1
if [[ "${SLURM_JOB_PARTITION}" == "t4v2" ]] || \
    [[ "${SLURM_JOB_PARTITION}" == "rtx6000" ]]; then
    export NCCL_SOCKET_IFNAME=bond0
fi

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

# lightning should figure out ranks and world size via the slurm env vars
# world size = SLURM_NTASKS
# global rank = SLURM_PROCID
# local rank = SLURM_LOCALID
# also need to explicitly set nnodes on srun since it seems to default to 1

cd ../scripts
srun --nodes=$SLURM_JOB_NUM_NODES python train.py -c ldm.yaml --num_nodes $SLURM_JOB_NUM_NODES --num_devices $SLURM_NTASKS_PER_NODE

wait
