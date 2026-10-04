#!/bin/sh
pkill -f "bin/radianc[e]" && echo "stopping" || echo "not running"
while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done
echo "stopped"
