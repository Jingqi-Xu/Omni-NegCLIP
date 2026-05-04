#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --account=pabeerel_971
#SBATCH --mail-type=all
#SBATCH --mail-user=jingqixu@usc.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load conda

eval "$(conda shell.bash hook)"
conda activate omninegclip
cd src


    

python training_omni_negclip.py \
    --clip-model-name=ViT-B/32 \
    --experiment-name=omni_negclip \
    --json-path=/project2/pabeerel_971/LucaXu/CoN-CLIP/annotations/negationclip_captions_train2014.json  \
    --image-dir=/project2/pabeerel_971/LucaXu/CoN-CLIP/train2014 \
    --trainable-layers front \
    --lambda-text 1.0 \
    --lambda-stage2 1.0 \
    --margin 0.9 \
    --epochs=30 \
    --batch-size=128
    
