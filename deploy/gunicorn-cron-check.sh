#!/usr/bin/env bash
# Simple Gunicorn monitor - check every 30 seconds
# Add to crontab: */30 * * * * $SCRIPT_DIR/gunicorn-cron-check.sh >> $LOG_DIR/cron-monitor.log 2>&1

APP_NAME="botCtl"
PID_FILE="${HOME}/.gunicorn/${APP_NAME}.pid"
APP_URL="${APP_URL:-http://0.0.0.0:11354}"
SCRIPT_DIR="${HOME}/auenBot/deploy"
LOG_DIR="${HOME}/auenBot/logs"
HINT_URL="${APP_URL}/api"

# Check health
curl -sf --connect-timeout 5 --max-time 10 "${HINT_URL}" > /dev/null 2>&1
status=$?

if [ $status -eq 0 ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') - OK - Health check passed"
else
  echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING - Restarting process"
  $SCRIPT_DIR/start-gunicorn.sh app > /dev/null 2>&1
  echo "$(date '+%Y-%m-%d %H:%M:%S') - Started with new PID: $(cat $PID_FILE 2>/dev/null)"
  exit 1
fi

