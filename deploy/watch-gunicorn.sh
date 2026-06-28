#!/usr/bin/env bash
# Manual watchdog: monitor Gunicorn and restart if dead
# Usage: ./watch-gunicorn.sh &

APP_NAME="botCtl"
PID_FILE="${HOME}/.gunicorn/${APP_NAME}.pid"
LOG_DIR="${HOME}/auenBot/logs"
APP_DIR="${HOME}/auenBot/botCtl"
APP_URL="${APP_URL:-http://0.0.0.0:11354}"
SCRIPT_DIR="${HOME}/auenBot/deploy"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG_DIR}/watchdog.log"
}

log "Starting manual watchdog..."

count=0
while true; do
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$PID" ] && [ -d "/proc/$PID" ]; then
            log "Gunicorn running (PID: $PID)"
        else
            log "Gunicorn died - restarting..."
            SCRIPT_DIR/start-gunicorn.sh app > /dev/null 2>&1
            log "Started with new PID: $(cat $PID_FILE 2>/dev/null)"
        fi
    else
        log "No PID file - starting Gunicorn..."
        SCRIPT_DIR/start-gunicorn.sh app > /dev/null 2>&1
    fi
    
    # Health check every 30 seconds
    count=$((count + 1))
    if [ $((count % 6)) -eq 0 ]; then
        curl -sf --connect-timeout 2 --max-time 5 "$APP_URL/health" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            log "Health check OK"
        else
            log "Health check failed"
        fi
    fi
    
    sleep 5
done

