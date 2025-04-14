#!/bin/bash

# Set the required variables.
bids_root_dir="/data/s.dharia-ra/PEARL/PEARL/bids_dataset"
derivatives_dir="${bids_root_dir}/derivatives"
fs_license="/data/s.dharia-ra/PEARL/PEARL/license.txt"
nthreads=64         # Adjust as needed
mem_mb=20000      # Adjust as needed

# Loop over each subject folder (assumes folders are named sub-XX)
for subj_path in ${bids_root_dir}/sub-*; do
  subj=$(basename "$subj_path")
  echo "Running fMRIPrep for subject: ${subj}"
  
  fmriprep-docker $bids_root_dir $derivatives_dir \
    participant \
    --participant-label $subj \
    --skip-bids-validation \
    --md-only-boilerplate \
    --fs-license-file $fs_license \
    --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --mem_mb $mem_mb \
    --bold2anat-init t1w
done
