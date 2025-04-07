#!/bin/bash
# P160 Parallel Search with improved log-based status and advanced strategies

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

# Set up paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_FILE="$SCRIPT_DIR/p160_status.log"

# Initialize log file
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Starting P160 search ($DURATION hours)" > $LOG_FILE

# Function to check if processes are still running
check_processes() {
    ACTIVE_COUNT=0
    for PID in $PID1 $PID2 $PID3 $PID4; do
        if ps -p $PID > /dev/null; then
            ACTIVE_COUNT=$((ACTIVE_COUNT + 1))
        fi
    done
}

# Function to log status
log_status() {
    # Get status from each process
    check_processes
    echo "[$(date +%H:%M:%S)] Running: $(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS) | Processes: $ACTIVE_COUNT/4" >> $LOG_FILE
}

# Function to find and share the best match between all strategies
share_best_match() {
    BEST_MATCH=0
    BEST_DIR=""
    
    # Find the best match across all strategies
    for DIR in strategy1 strategy2 strategy3 strategy4; do
        if [ -f "$DIR/best_match.txt" ]; then
            MATCH=$(grep "Match percent:" "$DIR/best_match.txt" 2>/dev/null | awk '{print $3}' | tr -d '%')
            if [[ "$MATCH" =~ ^[0-9]+$ ]] && [ "$MATCH" -gt "$BEST_MATCH" ]; then
                BEST_MATCH=$MATCH
                BEST_DIR=$DIR
            fi
        fi
    done
    
    # If we found a best match, copy it to all other directories
    if [ -n "$BEST_DIR" ] && [ -f "$BEST_DIR/best_match.txt" ]; then
        # Store the last logged match to check if this is a new one
        LAST_LOGGED_MATCH=$(grep "Match percent:" best_match.txt 2>/dev/null | awk '{print $3}' | tr -d '%')
        LAST_LOGGED_INPUT=$(grep "Input:" best_match.txt 2>/dev/null | awk '{print $2}')
        
        # Copy best match to root directory
        cp "$BEST_DIR/best_match.txt" ./best_match.txt.new
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
        # Strategy 1: All advanced features
        ./p160_simple --attempts 5000000 --batches 20 --guided-search --pattern-learning --adaptive-mutation --load > s1_output.log 2>&1
        sleep 1
    done
) > /dev/null 2>&1 &
PID1=$!
cd "$SCRIPT_DIR"

cd strategy2
(
    while [ $(date +%s) -lt $END_TIME ]; do
        # Strategy 2: Guided search + adaptive mutation
        ./p160_simple --attempts 10000000 --batches 20 --guided-search --adaptive-mutation --load > s2_output.log 2>&1
        sleep 1
    done
) > /dev/null 2>&1 &
PID2=$!
cd "$SCRIPT_DIR"

cd strategy3
(
    while [ $(date +%s) -lt $END_TIME ]; do
        # Strategy 3: Pattern learning
        ./p160_simple --attempts 12500000 --batches 10 --pattern-learning --load > s3_output.log 2>&1
        sleep 1
    done
) > /dev/null 2>&1 &
PID3=$!
cd "$SCRIPT_DIR"

cd strategy4
(
    while [ $(date +%s) -lt $END_TIME ]; do
        # Strategy 4: Adaptive mutation + random
        ./p160_simple --attempts 10000000 --batches 10 --adaptive-mutation --random --load > s4_output.log 2>&1
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
