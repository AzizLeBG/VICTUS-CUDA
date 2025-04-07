#!/bin/bash
# Analyze matching bytes in best results

# Target hash
TARGET="0385663c8b2f90659e1ccab201694f4f8ec24b3749cfe5030c7c3646a709408e19"

# Check if best_match.txt exists
if [ ! -f "best_match.txt" ]; then
    echo "No best_match.txt found."
    exit 1
fi

# Extract the hash from best_match.txt
BEST_HASH=$(grep "Hash:" best_match.txt | cut -d' ' -f2)
BEST_INPUT=$(grep "Input:" best_match.txt | cut -d' ' -f2)
BEST_PERCENT=$(grep "Match percent:" best_match.txt | cut -d' ' -f3 | tr -d '%')

echo "Analyzing byte matching patterns:"
echo "Current best match: $BEST_PERCENT%"
echo "Input: $BEST_INPUT"
echo "Hash:  $BEST_HASH"
echo "Target: $TARGET"
echo ""

# Compare byte by byte
echo "Byte-by-byte comparison:"
echo "Pos | Hash | Target | Match"
echo "-----------------------"
MATCHES=0
for i in {0..31}; do
    HASH_BYTE="${BEST_HASH:$((i*2)):2}"
    TARGET_BYTE="${TARGET:$((i*2)):2}"
    
    if [ "$HASH_BYTE" == "$TARGET_BYTE" ]; then
        MATCH="✓"
        MATCHES=$((MATCHES + 1))
    else
        MATCH="✗"
    fi
    
    printf "%2d  | %s  | %s   | %s\n" $i $HASH_BYTE $TARGET_BYTE "$MATCH"
done

echo ""
echo "Total matching bytes: $MATCHES/32 ($(($MATCHES * 100 / 32))%)"
echo ""
echo "Strategy recommendations:"
if [ $MATCHES -lt 8 ]; then
    echo "- Early stage: Focus on random exploration"
    echo "- Consider varying bytes 0-7 and 24-31 more frequently"
elif [ $MATCHES -lt 16 ]; then
    echo "- Mid stage: Focus on matched bytes"
    echo "- Consider locking matched bytes and varying others"
else
    echo "- Late stage: Very promising match"
    echo "- Consider systematic exploration of remaining unmatched bytes"
fi
