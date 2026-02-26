#!/bin/bash
# Monitor experiment progress, check every 10 minutes, 100 iterations
LOG="/root/data1/jinhyeok/LangSplat/log/experiment/monitor_v2.log"
cd /root/data1/jinhyeok/LangSplat

for i in $(seq 1 100); do
    echo "=== Monitor check #$i at $(date) ===" >> "$LOG"
    
    # Check if main script is still running
    if ! ps aux | grep -v grep | grep "run_experiment_resume" > /dev/null 2>&1; then
        echo "EXPERIMENT SCRIPT HAS FINISHED" >> "$LOG"
        echo "Final timing results:" >> "$LOG"
        cat log/experiment/timing_results.txt >> "$LOG" 2>/dev/null
        echo "Final eval results:" >> "$LOG"
        cat log/experiment/eval_results.txt >> "$LOG" 2>/dev/null
        break
    fi
    
    # Check what's currently running
    CURRENT=$(ps aux | grep python | grep -v grep | grep -oP '(?<=python ).*' | head -1)
    echo "Currently running: $CURRENT" >> "$LOG"
    
    # Check latest progress from most recent log file
    LATEST_LOG=$(ls -t log/experiment/3d_ovs/*.log log/experiment/dl3dv/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        PROGRESS=$(grep -oP '\d+/30000' "$LATEST_LOG" 2>/dev/null | tail -1)
        echo "Latest log: $LATEST_LOG, Progress: $PROGRESS" >> "$LOG"
    fi
    
    # Check for errors in most recent log
    LATEST_LOG_ALL=$(ls -t log/experiment/3d_ovs/*.log log/experiment/dl3dv/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG_ALL" ]; then
        ERRORS=$(grep -i "error\|traceback\|exception" "$LATEST_LOG_ALL" 2>/dev/null | tail -3)
        if [ -n "$ERRORS" ]; then
            echo "ERRORS DETECTED in $LATEST_LOG_ALL:" >> "$LOG"
            echo "$ERRORS" >> "$LOG"
        fi
    fi
    
    # Current results count
    TIMING_COUNT=$(wc -l < log/experiment/timing_results.txt 2>/dev/null || echo 0)
    EVAL_COUNT=$(wc -l < log/experiment/eval_results.txt 2>/dev/null || echo 0)
    echo "Timing results: $TIMING_COUNT scenes, Eval results: $EVAL_COUNT scenes" >> "$LOG"
    echo "" >> "$LOG"
    
    sleep 600
done
