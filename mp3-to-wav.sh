#!/usr/bin/bash
#SBATCH --job-name=mp3_to_wav_array
#SBATCH --output=mp3_to_wav_%A_%a.log
#SBATCH --error=mp3_to_wav_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH -p hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512MB
#SBATCH --array=0   # 558 tasks total if batching 100 files per task

# Load modules if needed
ml load system ffmpeg

SRC_DIR="$SCRATCH/mtg-jamendo"
DST_DIR="$SCRATCH/mtg-jamendo-wav"
mkdir -p "$DST_DIR"

# Batch size: number of files per array task
BATCH_SIZE=100000

# Find all MP3 files (one-time generation)
FILE_LIST="$TMPDIR/mp3_files.txt"
if [ ! -f "$FILE_LIST" ]; then
    find "$SRC_DIR" -type f -name "*.mp3" > "$FILE_LIST"
fi

TOTAL_FILES=$(wc -l < "$FILE_LIST")

# Compute start and end index for this array task
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1))
END=$((START + BATCH_SIZE - 1))
if [ $END -gt $TOTAL_FILES ]; then
    END=$TOTAL_FILES
fi

echo "Processing files $START to $END out of $TOTAL_FILES"

# Loop through files in this batch
for i in $(seq $START $END); do
    MP3_FILE=$(sed -n "${i}p" "$FILE_LIST")
    REL_PATH="${MP3_FILE#$SRC_DIR/}"
    DST_FILE="$DST_DIR/${REL_PATH%.mp3}.wav"
    mkdir -p "$(dirname "$DST_FILE")"

    # Only convert if destination file doesn't exist
    if [ ! -f "$DST_FILE" ]; then
        echo "Converting: $MP3_FILE -> $DST_FILE"
        ffmpeg -y -i "$MP3_FILE" -ar 44100 -ac 2 "$DST_FILE"
    else
        echo "Skipping existing file: $DST_FILE"
    fi
done

echo "Finished processing files $START to $END out of $TOTAL_FILES"
