#!/bin/bash

# Define paths
BIDS_ROOT="/home/s.dharia-ra/Shyamal/PEARL/bids_dataset"
OUTPUT_DIR="/home/s.dharia-ra/Shyamal/PEARL/topup_results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Loop through all subjects in the dataset
for SUBJECT in $(ls $BIDS_ROOT | grep "sub-"); do
    echo "Processing $SUBJECT..."

    # Define paths for AP and PA files
    AP_FILE="$BIDS_ROOT/$SUBJECT/func/${SUBJECT}_task-rest_dir-AP_bold.nii.gz"
    PA_FILE="$BIDS_ROOT/$SUBJECT/func/${SUBJECT}_task-rest_dir-PA_bold.nii.gz"

    # Check if both files exist
    if [[ ! -f $AP_FILE || ! -f $PA_FILE ]]; then
        echo "Missing AP or PA files for $SUBJECT. Skipping..."
        continue
    fi

    # (Optional) Get number of volumes in each file using fslnvols, just for reporting
    AP_VOLS=$(fslnvols $AP_FILE)
    PA_VOLS=$(fslnvols $PA_FILE)
    echo "$SUBJECT: AP has $AP_VOLS volumes, PA has $PA_VOLS volumes."

    # Compute the mean image along the time dimension for AP and PA
    AP_MEAN="$OUTPUT_DIR/${SUBJECT}_AP_mean.nii.gz"
    PA_MEAN="$OUTPUT_DIR/${SUBJECT}_PA_mean.nii.gz"
    echo "Calculating mean image for AP..."
    fslmaths $AP_FILE -Tmean $AP_MEAN
    echo "Calculating mean image for PA..."
    fslmaths $PA_FILE -Tmean $PA_MEAN

    # Merge the mean AP and PA images into a single 4D file for TOPUP
    MERGED_MEAN_FILE="$OUTPUT_DIR/${SUBJECT}_mean_AP_PA_merged.nii.gz"
    echo "Merging mean AP and PA images into $MERGED_MEAN_FILE..."
    fslmerge -t $MERGED_MEAN_FILE $AP_MEAN $PA_MEAN

    # Create acqparams file for TOPUP (2 volumes: first line for AP, second for PA)
    ACQPARAMS_FILE="$OUTPUT_DIR/${SUBJECT}_mean_acqparams.txt"
    echo "Creating acqparams.txt for the mean volumes for $SUBJECT..."
    > $ACQPARAMS_FILE  # Clear the file if it exists
    echo "0 -1 0 0.059999" >> $ACQPARAMS_FILE
    echo "0 1 0 0.059999" >> $ACQPARAMS_FILE

    # Run TOPUP using the merged mean image
    TOPUP_BASE="$OUTPUT_DIR/${SUBJECT}_mean_topup_results"
    echo "Running TOPUP on mean images for $SUBJECT..."
    topup --imain=$MERGED_MEAN_FILE --datain=$ACQPARAMS_FILE --config=b02b0.cnf --out=$TOPUP_BASE --fout=${TOPUP_BASE}_fout --iout=${TOPUP_BASE}_iout --nthr=64

    echo "TOPUP field estimation completed for $SUBJECT."

    # For separate applytopup outputs, create two separate acqparams files,
    # each with a single line corresponding to the appropriate phase-encode direction.

    # acqparams for the AP file (1-line)
    APPLYTOPUP_ACQPARAMS_AP="$OUTPUT_DIR/${SUBJECT}_applytopup_acqparams_AP.txt"
    echo "Creating applytopup acqparams file for AP for $SUBJECT..."
    > $APPLYTOPUP_ACQPARAMS_AP
    echo "0 -1 0 0.059999" >> $APPLYTOPUP_ACQPARAMS_AP

    # acqparams for the PA file (1-line)
    APPLYTOPUP_ACQPARAMS_PA="$OUTPUT_DIR/${SUBJECT}_applytopup_acqparams_PA.txt"
    echo "Creating applytopup acqparams file for PA for $SUBJECT..."
    > $APPLYTOPUP_ACQPARAMS_PA
    echo "0 1 0 0.059999" >> $APPLYTOPUP_ACQPARAMS_PA

    # Now, run applytopup separately for the AP and PA files to generate separate outputs.

    # Apply TOPUP correction for AP
    APPLYTOPUP_OUT_AP="$OUTPUT_DIR/${SUBJECT}_undistorted_AP.nii.gz"
    echo "Applying TOPUP correction to AP data for $SUBJECT..."
    applytopup --imain=$AP_FILE \
               --datain=$APPLYTOPUP_ACQPARAMS_AP \
               --inindex=1 \
               --topup=$TOPUP_BASE \
               --out=$APPLYTOPUP_OUT_AP \
               --method=jac
    echo "ApplyTOPUP for AP completed for $SUBJECT. Undistorted AP image saved at $APPLYTOPUP_OUT_AP."

    # Apply TOPUP correction for PA
    APPLYTOPUP_OUT_PA="$OUTPUT_DIR/${SUBJECT}_undistorted_PA.nii.gz"
    echo "Applying TOPUP correction to PA data for $SUBJECT..."
    applytopup --imain=$PA_FILE \
               --datain=$APPLYTOPUP_ACQPARAMS_PA \
               --inindex=1 \
               --topup=$TOPUP_BASE \
               --out=$APPLYTOPUP_OUT_PA \
               --method=jac
    echo "ApplyTOPUP for PA completed for $SUBJECT. Undistorted PA image saved at $APPLYTOPUP_OUT_PA."

    #### Copy (or move) the final files into each subject’s func folder ####
    DESTINATION_DIR="$BIDS_ROOT/$SUBJECT/func"
    echo "Copying final TOPUP outputs to $DESTINATION_DIR..."
    mkdir -p $DESTINATION_DIR

    # Copy final TOPUP and APPLYTOPUP outputs
    cp $APPLYTOPUP_OUT_AP $DESTINATION_DIR/
    cp $APPLYTOPUP_OUT_PA $DESTINATION_DIR/

    # # Optionally, also copy the TOPUP field maps and merged mean image
    # cp ${TOPUP_BASE}_fout.nii.gz $DESTINATION_DIR/
    # cp ${TOPUP_BASE}_iout.nii.gz $DESTINATION_DIR/
    # cp $MERGED_MEAN_FILE $DESTINATION_DIR/
    # cp $ACQPARAMS_FILE $DESTINATION_DIR/

    # # Also copy the applytopup acqparams files if needed
    # cp $APPLYTOPUP_ACQPARAMS_AP $DESTINATION_DIR/
    # cp $APPLYTOPUP_ACQPARAMS_PA $DESTINATION_DIR/

    echo "Final files for $SUBJECT have been copied to $DESTINATION_DIR."
    
done

echo "All subjects processed!"
