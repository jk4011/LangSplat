#!/bin/bash
# Monitor experiment every 10 minutes, 100 iterations
LOG_DIR="/root/data1/jinhyeok/LangSplat/log/experiment"
MONITOR_LOG="/root/data1/jinhyeok/LangSplat/log/experiment/monitor.log"

for i in $(seq 1 100); do
    sleep 600
    echo "========================================" >> "$MONITOR_LOG"
    echo "=== Monitor check #${i} at $(date) ===" >> "$MONITOR_LOG"
    
    # Check if process is still running
    if ! ps aux | grep -v grep | grep "run_experiment_resume" > /dev/null 2>&1; then
        echo "PROCESS ENDED" >> "$MONITOR_LOG"
        # Check if it completed or errored
        echo "--- Timing results ---" >> "$MONITOR_LOG"
        cat "$LOG_DIR/timing_results.txt" >> "$MONITOR_LOG" 2>/dev/null
        echo "" >> "$MONITOR_LOG"
        echo "--- Eval results ---" >> "$MONITOR_LOG"
        cat "$LOG_DIR/eval_results.txt" >> "$MONITOR_LOG" 2>/dev/null
        echo "" >> "$MONITOR_LOG"
        
        # Check for Experiment.md
        if [ -f "/root/data1/jinhyeok/LangSplat/Experiment.md" ]; then
            echo "Experiment.md EXISTS - script completed successfully" >> "$MONITOR_LOG"
        else
            echo "Experiment.md NOT FOUND - script may have errored" >> "$MONITOR_LOG"
            # Find the latest log file with errors
            LATEST_LOG=$(ls -t "$LOG_DIR"/{lerf,3d_ovs,dl3dv}/*.log 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                echo "Latest log: $LATEST_LOG" >> "$MONITOR_LOG"
                tail -20 "$LATEST_LOG" >> "$MONITOR_LOG" 2>/dev/null
            fi
        fi
        echo "MONITORING COMPLETE" >> "$MONITOR_LOG"
        break
    fi
    
    # Get current progress
    echo "--- Timing results so far ---" >> "$MONITOR_LOG"
    cat "$LOG_DIR/timing_results.txt" >> "$MONITOR_LOG" 2>/dev/null
    echo "" >> "$MONITOR_LOG"
    echo "--- Eval results so far ---" >> "$MONITOR_LOG"
    cat "$LOG_DIR/eval_results.txt" >> "$MONITOR_LOG" 2>/dev/null
    echo "" >> "$MONITOR_LOG"
    
    # Check latest log activity
    echo "--- Latest log files ---" >> "$MONITOR_LOG"
    ls -lt "$LOG_DIR"/{lerf,3d_ovs,dl3dv}/*.log 2>/dev/null | head -3 >> "$MONITOR_LOG"
    echo "" >> "$MONITOR_LOG"
    
    # Check for errors in recent logs
    LATEST_LOG=$(ls -t "$LOG_DIR"/{lerf,3d_ovs,dl3dv}/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        if grep -q "Error\|Traceback\|FAILED" "$LATEST_LOG" 2>/dev/null; then
            echo "!!! ERROR DETECTED in $LATEST_LOG !!!" >> "$MONITOR_LOG"
            grep -A5 "Error\|Traceback\|FAILED" "$LATEST_LOG" >> "$MONITOR_LOG" 2>/dev/null
        else
            # Show current iteration from last line
            LAST_LINE=$(tail -1 "$LATEST_LOG" 2>/dev/null)
            ITER=$(echo "$LAST_LINE" | grep -oP "(\d+)/30000" | tail -1)
            if [ -n "$ITER" ]; then
                echo "Current: $LATEST_LOG at iteration $ITER" >> "$MONITOR_LOG"
            else
                echo "Current: $LATEST_LOG" >> "$MONITOR_LOG"
                echo "Last line: $(echo "$LAST_LINE" | tail -c 200)" >> "$MONITOR_LOG"
            fi
        fi
    fi
    echo "" >> "$MONITOR_LOG"
done
