#!/usr/bin/env python3
"""GPU monitoring agent - runs on H100 server, pushes to dashboard."""

import json
import os
import re
import socket
import subprocess
import threading
import time
from datetime import datetime

import psutil
import pynvml
import requests

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://your-dashboard-host:8080/update")
INTERVAL = 5  # seconds
HOSTNAME = socket.gethostname()

# XID error meanings (most common ones)
XID_DESC = {
    "8":   "GPU memory access fault",
    "31":  "GPU memory page fault",
    "48":  "Double-bit ECC error",
    "61":  "Internal micro-controller halt (deadlock)",
    "62":  "Internal micro-controller halt (deadlock)",
    "63":  "ECC page retirement",
    "74":  "NVLink fatal error",
    "79":  "GPU fallen off the bus (hard hang)",
    "92":  "High single-bit ECC error rate",
    "119": "GSP RPC timeout (firmware unresponsive)",
    "154": "GPU recovery required (node reboot needed)",
}

event_log = []  # in-memory event log, newest first
event_lock = threading.Lock()


def add_event(level, msg, gpu_id=None, processes=None):
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,   # "error" / "warn" / "info"
        "msg": msg,
        "gpu_id": gpu_id,
        "processes": processes or [],
    }
    with event_lock:
        event_log.insert(0, entry)
        if len(event_log) > 200:
            event_log.pop()
    gpu_tag = f" [GPU{gpu_id}]" if gpu_id is not None else ""
    print(f"[EVENT] {entry['time']} [{level.upper()}]{gpu_tag} {msg}")


def get_cuda_visible(pid):
    """Read CUDA_VISIBLE_DEVICES from a process's environ (root can read any).

    Captures the target GPU(s) at launch time, independent of the 5s
    nvidia-smi poll — so even a process that crashes within seconds still
    gets attributed. Returns the raw value (e.g. "2" or "0,1" or a UUID),
    or None if unset/unreadable.
    """
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            data = f.read()
        for kv in data.split(b"\x00"):
            if kv.startswith(b"CUDA_VISIBLE_DEVICES="):
                return kv.split(b"=", 1)[1].decode(errors="replace")
    except Exception:
        pass
    return None


def get_gpu_processes():
    """Return {gpu_id: [{"user", "pid", "cmd", "mem_mb"}]}"""
    result = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            timeout=5
        ).decode()
    except Exception:
        return result

    # build uuid -> index map
    uuid_to_idx = {}
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid_to_idx[pynvml.nvmlDeviceGetUUID(h)] = i
    except Exception:
        pass

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
        gpu_idx = uuid_to_idx.get(uuid, -1)
        try:
            pid = int(pid_s)
            mem_mb = int(mem_s)
            proc = psutil.Process(pid)
            user = proc.username()
            cmd = " ".join(proc.cmdline())[:80]
            entry = {"user": user, "pid": pid, "cmd": cmd, "mem_mb": mem_mb}
            result.setdefault(gpu_idx, []).append(entry)
        except Exception:
            pass
    return result


def get_gpu_stats(gpu_procs):
    """Return list of GPU stat dicts."""
    stats = []
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return stats

    for i in range(count):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util_gpu = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except pynvml.NVMLError:
                util_gpu = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = None
            try:
                ecc = pynvml.nvmlDeviceGetTotalEccErrors(
                    h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_AGGREGATE_ECC)
            except pynvml.NVMLError:
                ecc = 0
            stats.append({
                "id": i,
                "util": util_gpu,
                "mem_used_gb": round(mem.used / 1024**3, 1),
                "mem_total_gb": round(mem.total / 1024**3, 1),
                "temp": temp,
                "ecc_errors": ecc,
                "status": "ok",
                "processes": gpu_procs.get(i, []),
            })
        except pynvml.NVMLError as e:
            stats.append({
                "id": i, "status": "error",
                "error": str(e),
                "processes": gpu_procs.get(i, []),
            })
    return stats


def check_gpu_hang():
    """Try nvidia-smi with tight timeout; flag hung GPUs."""
    hung = []
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            timeout=4
        )
    except subprocess.TimeoutExpired:
        hung = list(range(6))  # all unresponsive
        procs = _snapshot_gpu_procs_cached()
        add_event("error", "nvidia-smi timed out — one or more GPUs may be hung", processes=procs)
    except Exception:
        pass
    return hung


_last_gpu_procs = []
_known_pids: set[int] = set()  # track seen PIDs to detect start/stop


def _snapshot_gpu_procs_cached():
    return _last_gpu_procs


def detect_process_changes(flat: list[dict]):
    """Emit events when GPU processes start or stop."""
    global _known_pids
    current = {p["pid"]: p for p in flat}
    current_pids = set(current.keys())

    for pid in current_pids - _known_pids:
        p = current[pid]
        cvd = get_cuda_visible(pid)
        cvd_tag = f" (CUDA_VISIBLE_DEVICES={cvd})" if cvd is not None else ""
        add_event(
            "info",
            f"{p['user']} started: {p['cmd']}{cvd_tag}",
            gpu_id=p["gpu_id"],
            processes=[p],
        )

    for pid in _known_pids - current_pids:
        add_event("info", f"PID {pid} finished (process exited)")

    _known_pids = current_pids


def watch_dmesg():
    """Tail dmesg in a background thread and capture XID / NVRM errors."""
    try:
        proc = subprocess.Popen(
            ["dmesg", "--follow", "--level=err,warn"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True
        )
    except Exception:
        # fallback: sudo dmesg -w
        try:
            proc = subprocess.Popen(
                ["sudo", "dmesg", "-w"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True
            )
        except Exception:
            print("[WARN] Cannot tail dmesg — XID detection disabled")
            return

    # build PCI bus -> GPU index map for XID attribution
    pci_to_idx = {}
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            pci = pynvml.nvmlDeviceGetPciInfo(h)
            bus = f"{pci.domain:04x}:{pci.bus:02x}:{pci.device:02x}"
            pci_to_idx[bus] = i
    except Exception:
        pass

    # matches: Xid (PCI:0000:40:00): 119
    xid_pattern = re.compile(r"Xid\s*\(PCI:([0-9a-f:]+)\):\s*(\d+)", re.I)
    gpu_pattern = re.compile(r"GPU(\d+)", re.I)

    for line in proc.stdout:
        line = line.strip()
        if "NVRM" not in line and "nvidia" not in line.lower():
            continue

        xid_match = xid_pattern.search(line)
        if xid_match:
            pci_addr = xid_match.group(1)   # e.g. "0000:40:00"
            xid = xid_match.group(2)
            desc = XID_DESC.get(xid, "unknown error")
            gpu_id = pci_to_idx.get(pci_addr)
            if gpu_id is None:
                # fallback: look for "GPU<N>" mention in message
                gm = gpu_pattern.search(line)
                gpu_id = int(gm.group(1)) if gm else None
            level = "error" if xid in ("61", "62", "79", "48", "119", "154") else "warn"
            procs = _snapshot_gpu_procs_cached()
            if gpu_id is not None:
                procs = [p for p in procs if p.get("gpu_id") == gpu_id]
            add_event(level, f"XID {xid}: {desc} | raw: {line[:120]}", gpu_id=gpu_id, processes=procs)
        elif "error" in line.lower() or "fault" in line.lower():
            add_event("warn", f"NVRM: {line[:150]}")


def push(payload):
    try:
        requests.post(DASHBOARD_URL, json=payload, timeout=4)
    except Exception as e:
        print(f"[WARN] push failed: {e}")


def main():
    print(f"[agent] starting, pushing to {DASHBOARD_URL} every {INTERVAL}s")
    threading.Thread(target=watch_dmesg, daemon=True).start()

    while True:
        gpu_procs = get_gpu_processes()
        # flatten for snapshot
        flat = []
        for gid, ps in gpu_procs.items():
            for p in ps:
                flat.append({**p, "gpu_id": gid})
        global _last_gpu_procs
        _last_gpu_procs = flat
        detect_process_changes(flat)

        gpu_stats = get_gpu_stats(gpu_procs)
        check_gpu_hang()

        with event_lock:
            recent_events = event_log[:50]

        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        payload = {
            "hostname": HOSTNAME,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_pct": cpu_pct,
            "mem_pct": mem.percent,
            "mem_used_gb": round(mem.used / 1024**3, 1),
            "mem_total_gb": round(mem.total / 1024**3, 1),
            "gpus": gpu_stats,
            "events": recent_events,
        }
        push(payload)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
