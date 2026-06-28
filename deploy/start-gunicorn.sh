#!/usr/bin/env bash
# Start Gunicorn for Flask app (user mode)
# Usage: ./start-gunicorn.sh [module_name]
# Example: ./start-gunicorn.sh app

APP_NAME="botCtl"
PID_FILE="${HOME}/.gunicorn/${APP_NAME}.pid"
LOG_DIR="${HOME}/auenBot/logs"
APP_DIR="${HOME}/auenBot/botCtl/"
SCRIPT_DIR="${HOME}/auenBot/deploy"

mkdir -p "$LOG_DIR" "${HOME}/.gunicorn"

cd "$APP_DIR"

echo "[INFO] Starting Gunicorn for ${APP_NAME}:app (PID: $PID_FILE)"

/usr/bin/nohup /usr/bin/gunicorn --pid "$PID_FILE" --access-logfile "${LOG_DIR}/gunicorn-access.log" --error-logfile "${LOG_DIR}/gunicorn-error.log"  --workers 2 --bind 0.0.0.0:11354  "${APP_NAME}:app" > /dev/null 2>&1 &

echo "Gunicorn PID: $!"

sleep 3

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Gunicorn started, PID=$(cat $PID_FILE)"
    exit 0
else
    echo "[ERROR] Gunicorn failed to start"
    exit 1
fi


