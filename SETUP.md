# GPU Monitor — Setup Guide

Two components:
- **Agent** — runs on every GPU/compute server, pushes metrics to dashboard
- **Dashboard** — runs on one machine (e.g. mini2), serves the web UI

---

## Dashboard Server (mini2 or any always-on machine)

### Install
```bash
pip3 install fastapi uvicorn
```

### Copy server file
```bash
scp server.py user@<dashboard-host>:~/gpu_monitor_server.py
```

### Run manually
```bash
python3 ~/gpu_monitor_server.py 8080
```

### Run as systemd service
```bash
sudo tee /etc/systemd/system/gpu-monitor-dashboard.service > /dev/null << 'EOF'
[Unit]
Description=GPU Monitor Dashboard
After=network.target

[Service]
Type=simple
User=<your-user>
ExecStart=/usr/bin/python3 /home/<your-user>/gpu_monitor_server.py 8080
Restart=always
RestartSec=10
StandardOutput=append:/home/<your-user>/gpu_monitor_dashboard.log
StandardError=append:/home/<your-user>/gpu_monitor_dashboard.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gpu-monitor-dashboard
sudo systemctl start gpu-monitor-dashboard
```

### Access
Open `http://<dashboard-host>:8080` in browser.

---

## Agent (each GPU/compute server)

### Requirements
- Python env with: `pynvml`, `psutil`, `requests`
- Root access (needed to read dmesg for XID errors)
- Network access to the dashboard host

### Install dependencies
```bash
# If using conda:
conda activate <your-env>
pip install pynvml psutil requests

# Or system python:
pip3 install pynvml psutil requests
```

### Configure
Edit `agent.py` line 17 to point to your dashboard:
```python
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://<dashboard-host>:8080/update")
```
Or set the env var instead (preferred for systemd).

### Run as systemd service

1. Copy agent to the server:
```bash
scp agent.py root@<compute-server>:/opt/gpu-monitor/agent.py
```

2. Create the service (adjust paths as needed):
```bash
sudo tee /etc/systemd/system/gpu-monitor-agent.service > /dev/null << 'EOF'
[Unit]
Description=GPU Monitor Agent
After=network.target

[Service]
Type=simple
User=root
ExecStart=/path/to/python3 /opt/gpu-monitor/agent.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/gpu-monitor-agent.log
StandardError=append:/var/log/gpu-monitor-agent.log
Environment=DASHBOARD_URL=http://<dashboard-host>:8080/update

[Install]
WantedBy=multi-user.target
EOF
```

3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gpu-monitor-agent
sudo systemctl start gpu-monitor-agent
```

4. Check status:
```bash
sudo systemctl status gpu-monitor-agent
sudo journalctl -u gpu-monitor-agent -f
```

### If using conda (no system python):
Replace `ExecStart` with:
```
ExecStart=/home/<user>/miniconda3/bin/conda run -n <env-name> python3 /opt/gpu-monitor/agent.py
```

---

## Adding a new server

1. Copy `agent.py` to the new server
2. Install dependencies (`pynvml psutil requests`)
3. Set `DASHBOARD_URL` to point to your dashboard
4. Run with root (for dmesg access)
5. The server will automatically appear in the dashboard sidebar

---

## What gets tracked

| Event | How detected |
|-------|-------------|
| User started GPU process | nvidia-smi process list diff |
| Process finished | nvidia-smi process list diff |
| GPU deadlock | dmesg XID 61/62 |
| GPU fallen off bus | dmesg XID 79 |
| GSP firmware timeout | dmesg XID 119 |
| GPU hang | nvidia-smi query timeout (>4s) |
| ECC errors | NVML query |

---

## XID Error Reference

| XID | Meaning | Severity |
|-----|---------|----------|
| 8 | GPU memory access fault | warn |
| 31 | GPU memory page fault | warn |
| 48 | Double-bit ECC error | error |
| 61 | Internal deadlock | error |
| 62 | Internal deadlock | error |
| 63 | ECC page retirement | warn |
| 74 | NVLINK error | warn |
| 79 | GPU fallen off bus | error |
| 92 | High ECC error rate | warn |
| 119 | GSP RPC timeout | error |
