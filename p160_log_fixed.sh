#!/bin/bash
# P160 Parallel Search with improved log-based status

# Default parameters
DURATION=12  # hours to run

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prepare log file
LOG_FILE="p160_status.log"
echo "P160 Search started at $(date)" > $LOG_FILE
echo "Running for $DURATION hours" >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Create directories for each strategy
mkdir -p strategy1 strategy2 strategy3 strategy4

# Copy necessary files and compile
for dir in strategy1 strategy2 strategy3 strategy4; do
    cp p160_simple.cu Makefile.simple $dir/ 2>/dev/null
    (cd $dir && make -f Makefile.simple >/dev/null 2>&1)
done

# Track the last logged match to avoid duplicate entries
LAST_LOGGED_MATCH=""
LAST_LOGGED_INPUT=""
LAST_LOGGED_HASH=""

# Function to log status update with change detection
log_status() {
    # Get best match info directly from each strategy directory
    BEST_MATCH=0
    BEST_DIR=""
    MATCH=""
    INPUT=""
    HASH=""
    
    for dir in strategy1 strategy2 strategy3 strategy4; do
        if [ -f "$dir/best_match.txt" ]; then
            DIR_MATCH=$(grep "Match percent:" "$dir/best_match.txt" 2>/dev/null | awk '{print $3}' | tr -d '%')
            if [[ "$DIR_MATCH" =~ ^[0-9]+$ ]] && [ "$DIR_MATCH" -gt "$BEST_MATCH" ]; then
                BEST_MATCH=$DIR_MATCH
                BEST_DIR=$dir
                MATCH=$(grep "Match percent:" "$dir/best_match.txt" 2>/dev/null | awk '{print $3}')
                INPUT=$(grep "Input:" "$dir/best_match.txt" 2>/dev/null | awk '{print $2}')
                HASH=$(grep "Hash:" "$dir/best_match.txt" 2>/dev/null | awk '{print $2}')
            fi
        fi
    done
    
    # Check if values have changed before logging
    if [ "$MATCH" != "$LAST_LOGGED_MATCH" ] || [ "$INPUT" != "$LAST_LOGGED_INPUT" ] || [ "$HASH" != "$LAST_LOGGED_HASH" ]; then
        echo "[$(date +%H:%M:%S)] Status update:" >> $LOG_FILE
        echo "  - Running time: $(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS) / $(printf "%02d:00:00" $DURATION)" >> $LOG_FILE
        echo "  - Active processes: $ACTIVE_COUNT/4" >> $LOG_FILE
        
        if [ -n "$MATCH" ]; then
            echo "  - Best match: $MATCH (found in $BEST_DIR)" >> $LOG_FILE
            echo "  - Input: $INPUT" >> $LOG_FILE
            echo "  - Hash: $HASH" >> $LOG_FILE
            
            # Update the tracking variables
            LAST_LOGGED_MATCH="$MATCH"
            LAST_LOGGED_INPUT="$INPUT"
            LAST_LOGGED_HASH="$HASH"
            
            # If this is a better match, also copy it to the main directory
            cp "$BEST_DIR/best_match.txt" ./ 2>/dev/null
        else
            echo "  - No matches found yet" >> $LOG_FILE
        fi
        
        echo "----------------------------------------" >> $LOG_FILE
    else
        # Just log a brief status without the unchanged match details
        echo "[$(date +%H:%M:%S)] Running: $(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS) | Processes: $ACTIVE_COUNT/4" >> $LOG_FILE
    fi
}

# Function to check if processes are running
check_processes() {
    ACTIVE_COUNT=0
    for pid in $PID1 $PID2 $PID3 $PID4; do
        if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
            ACTIVE_COUNT=$((ACTIVE_COUNT + 1))
        fi
    done
}

# Function to find and share best match directly from files
share_best_match() {
    BEST_MATCH=0
    BEST_DIR=""
    
    for dir in strategy1 strategy2 strategy3 strategy4; do
        if [ -f "$dir/best_match.txt" ]; then
            DIR_MATCH=$(grep "Match percent:" "$dir/best_match.txt" 2>/dev/null | awk '{print $3}' | tr -d '%')
            if [[ "$DIR_MATCH" =~ ^[0-9]+$ ]] && [ "$DIR_MATCH" -gt "$BEST_MATCH" ]; then
                BEST_MATCH=$DIR_MATCH
                BEST_DIR=$dir
            fi
        fi
    done
    
    if [ ! -z "$BEST_DIR" ]; then
        # Copy to all strategy directories
        for dir in strategy1 strategy2 strategy3 strategy4; do
            if [ "$dir" != "$BEST_DIR" ]; then
                cp "$BEST_DIR/best_match.txt" "$dir/best_match.txt.new" 2>/dev/null
                mv "$dir/best_match.txt.new" "$dir/best_match.txt" 2>/dev/null
            fi
        done
        
        # Copy to main directory for logging
        cp "$BEST_DIR/best_match.txt" ./best_match.txt.new 2>/dev/null
        mv ./best_match.txt.new ./best_match.txt 2>/dev/null
        
        # If this is a new best match, log it specifically
        NEW_MATCH=$(grep "Match percent:" "$BEST_DIR/best_match.txt" 2>/dev/null | awk '{print $3}')
        NEW_INPUT=$(grep "Input:" "$BEST_DIR/best_match.txt" 2>/dev/null | awk '{print $2}')
        NEW_HASH=$(grep "Hash:" "$BEST_DIR/best_match.txt" 2>/dev/null | awk '{print $2}')
        
        if [ "$NEW_MATCH" != "$LAST_LOGGED_MATCH" ] || [ "$NEW_INPUT" != "$LAST_LOGGED_INPUT" ]; then
            echo "[$(date +%H:%M:%S)] NEW BEST MATCH FOUND ($BEST_DIR):" >> $LOG_FILE
            cat "$BEST_DIR/best_match.txt" >> $LOG_FILE
            echo "----------------------------------------" >> $LOG_FILE
        fi
    fi
}

# Function to cleanup on exit
cleanup() {
    echo "[$(date +%H:%M:%S)] Search stopped." >> $LOG_FILE
    kill $PID1 $PID2 $PID3 $PID4 2>/dev/null
    echo "Search stopped. Check $LOG_FILE for final results."
    exit 0
}

# Set up trap for Ctrl+C
trap cleanup INT

# Calculate end time
END_TIME=$(($(date +%s) + $DURATION * 3600))

# Record start time
START_TIME=$(date +%s)

# Launch strategies in background with direct writing to best_match.txt
cd strategy1
(
    while [ $(date +%s) -lt $END_TIME ]; do
        # Run with output redirected to separate log
        ./p160_simple --attempts 5000000 --batches 20 --random --load > s1_output.log 2>&1
        sleep 1
    done
) > /dev/null 2>&1 &
PID1=$!
cd "$SCRIPT_DIR"

cd strategy2
(
    while [ $(date +%s) -lt $END_TIME ]; do
        # Run with output redirected to separate log
        ./p160_simple --attempts 10000000 --batches 20 --load > s2_output.log 2>&1
        sleep 1
    done
) > /dev/null 2>&1 &
PID2=$!
cd "$SCRIPT_DIR"

cd strategy3
(
    while [ $(date +%s) -lt $END_TIME ]; do
        # Run with output redirected to separate log
        ./p160_simple --attempts 12500000 --batches 10 --load > s3_output.log 2>&1
        sleep 1
    done
) > /dev/null 2>&1 &
PID3=$!
cd "$SCRIPT_DIR"

cd strategy4
(
    while [ $(date +%s) -lt $END_TIME ]; do
        # Run with output redirected to separate log
        ./p160_simple --attempts 10000000 --batches 10 --random --load > s4_output.log 2>&1
        sleep 1
    done
) > /dev/null 2>&1 &
PID4=$!
cd "$SCRIPT_DIR"

echo "All strategies launched in background. View progress with: tail -f $LOG_FILE"
echo "PIDs: $PID1 $PID2 $PID3 $PID4"

# Main loop - improved log-based approach
while [ $(date +%s) -lt $END_TIME ]; do
    # Calculate elapsed time
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    # Check if processes are still running
    check_processes
    
    # Share best match with reliable file operations
    share_best_match
    
    # Log status (every 30 seconds or when match changes)
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        log_status
    fi
    
    # Check for 100% match
    if [ -f "best_match.txt" ]; then
        MATCH=$(grep "Match percent:" best_match.txt 2>/dev/null | awk '{print $3}' | tr -d '%')
        if [[ "$MATCH" =~ ^[0-9]+$ ]] && [ "$MATCH" -eq 100 ]; then
            echo "[$(date +%H:%M:%S)] !!! SOLUTION FOUND !!!" >> $LOG_FILE
            cat best_match.txt >> $LOG_FILE
            echo "SOLUTION FOUND! Check $LOG_FILE for details."
            cleanup
        fi
    fi
    
    # If no processes are running, exit
    if [ $ACTIVE_COUNT -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] All processes have stopped." >> $LOG_FILE
        echo "All processes have stopped. Check $LOG_FILE for details."
        exit 1
    fi
    
    # Brief sleep
    sleep 1
done

# Time's up, kill all processes
echo "[$(date +%H:%M:%S)] Time's up ($DURATION hours)." >> $LOG_FILE
cleanup
