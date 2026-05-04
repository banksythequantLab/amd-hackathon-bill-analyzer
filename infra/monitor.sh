#!/usr/bin/env bash
# monitor.sh — rocm-smi idle watchdog
#
# Polls GPU utilization every 30s. If utilization is <5% for IDLE_THRESHOLD
# consecutive minutes, prints a loud warning and (optionally) auto-stops
# the instance.
#
# Usage:
#   ./infra/monitor.sh                 # watch only, never touch the instance
#   ./infra/monitor.sh --auto-stop     # actually power off after warning
#   IDLE_THRESHOLD=10 ./infra/monitor.sh  # custom idle threshold (minutes)
#
# Run in a tmux pane on the cloud instance so you don't bleed credits at idle.
# Each idle hour at $1.99/hr is real money now (no grace credits).

set -euo pipefail

IDLE_THRESHOLD="${IDLE_THRESHOLD:-5}"          # minutes
POLL_INTERVAL="${POLL_INTERVAL:-30}"           # seconds
UTIL_LOW="${UTIL_LOW:-5}"                       # % below this = idle
AUTO_STOP=false

for arg in "$@"; do
    case "$arg" in
        --auto-stop) AUTO_STOP=true ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

POLLS_PER_THRESHOLD=$(( IDLE_THRESHOLD * 60 / POLL_INTERVAL ))
idle_polls=0

echo "=== GPU idle watchdog ==="
echo "  Poll interval        : ${POLL_INTERVAL}s"
echo "  Idle threshold (min) : ${IDLE_THRESHOLD}"
echo "  Util cutoff          : <${UTIL_LOW}%"
echo "  Auto-stop            : ${AUTO_STOP}"
echo

while true; do
    # rocm-smi --showuse outputs include "GPU use (%): N"
    util=$(rocm-smi --showuse 2>/dev/null \
        | grep -Eo "GPU use \(%\): *[0-9]+" \
        | head -1 \
        | grep -Eo "[0-9]+$" \
        || echo "0")

    ts=$(date +"%H:%M:%S")

    if [[ "$util" -lt "$UTIL_LOW" ]]; then
        idle_polls=$((idle_polls + 1))
        remaining=$((POLLS_PER_THRESHOLD - idle_polls))
        if [[ $remaining -le 0 ]]; then
            echo
            echo "============================================="
            echo "[$ts] GPU IDLE for >${IDLE_THRESHOLD} minutes"
            echo "============================================="
            echo "  Util now: ${util}%"
            echo "  Per-hour cost: ~\$1.99"
            echo
            if $AUTO_STOP; then
                echo "[$ts] AUTO_STOP=true — powering off in 60s. Ctrl+C to abort."
                sleep 60
                sudo shutdown -h now
            else
                echo "[$ts] WARNING ONLY — pass --auto-stop to power off automatically."
                echo "[$ts] Resetting counter; will warn again in ${IDLE_THRESHOLD} min if still idle."
                idle_polls=0
            fi
            echo
        else
            echo "[$ts] util=${util}%  idle ${idle_polls}/${POLLS_PER_THRESHOLD} polls (${remaining} until warning)"
        fi
    else
        if [[ $idle_polls -gt 0 ]]; then
            echo "[$ts] util=${util}%  resumed (was idle ${idle_polls} polls)"
        fi
        idle_polls=0
    fi

    sleep "$POLL_INTERVAL"
done
