#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "===== Starting FaceCrime Backend Server ====="

# Start docker containers in detached mode
echo "Starting Docker containers..."
docker compose up -d

# Wait for backend server to be ready
echo "Waiting for backend server to be ready..."
sleep 10

# Check if the backend service is running and listening on port 8443
echo "Checking if backend service is running..."
if ! docker compose ps | grep -q "backend.*Up"; then
    echo "ERROR: Backend service failed to start. Check logs with 'docker compose logs backend'"
    exit 1
fi

# Enable Tailscale funnel for port 8443
echo "Setting up Tailscale funnel for port 8443..."
sudo tailscale funnel --bg 8443

# Verify Tailscale funnel is working
echo "Verifying Tailscale funnel status..."
TAILSCALE_STATUS=$(tailscale status --json | grep -o '"Funnel":[^,}]*')
if [[ $TAILSCALE_STATUS == *"true"* ]]; then
    echo "Tailscale funnel is active"
else
    echo "WARNING: Tailscale funnel might not be active. Check with 'tailscale status'"
fi

# Check if the port is actually listening
echo "Checking if port 8443 is listening..."
if sudo lsof -i -P -n | grep LISTEN | grep -q ":8443"; then
    echo "Port 8443 is listening"
else
    echo "WARNING: Port 8443 is not listening. The service might not be running correctly."
fi

# Display public URL
TAILSCALE_HOSTNAME=$(tailscale status --json | grep -o '"Self":{[^}]*' | grep -o '"DNSName":"[^"]*"' | cut -d'"' -f4 | sed 's/\.$//')
echo ""
echo "===== FaceCrime Backend Server is now running ====="
echo "Available on the internet at: https://${TAILSCALE_HOSTNAME}/"
echo "To stop the server: docker compose down"
echo "To view logs: docker compose logs -f"