#!/bin/bash

LOG_PATH="/data/public/system-tools/system.log"
TIMESTAMP="[$(date '+%Y-%m-%d %H:%M:%S')]"
TMP_LOG=$(mktemp)

# ========== GPU check (compute + fuser open-only) ==========
GPU_THRESHOLD=512
GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  # 非 root 时优先尝试无密码 sudo；失败就保持空串，后面会自动跳过 open-only 统计
  if sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
  fi
fi

for gpu_id in $GPU_IDS; do
    # 1) 真实在该 GPU 上有计算上下文的进程
    mapfile -t COMPUTE_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | grep -v '^$')
    COUNT=${#COMPUTE_PIDS[@]}
    breakdown_compute=""
    if [ "$COUNT" -gt 0 ]; then
        # 按用户聚合
        breakdown_compute=$(for pid in "${COMPUTE_PIDS[@]}"; do
            [ -d "/proc/$pid" ] || continue
            ps -o user= -p "$pid" 2>/dev/null
        done | sort | uniq -c | sort -nr | awk '{printf "%s(%s) ", $2, $1}')
    fi

    # 2) 仅打开了设备文件的进程（来自 fuser -v）
    breakdown_open=""
    if [ -n "$SUDO" ] || [ "$(id -u)" -eq 0 ]; then
        open_users=$($SUDO fuser -v /dev/nvidia"$gpu_id" 2>&1 \
            | awk '
                /^\/dev\/nvidia/ {next}      # 跳过设备名行
                /^USER/ {next}               # 跳过表头
                NF>=4 {print $1}             # 取用户名列
            ')
        if [ -n "$open_users" ]; then
            breakdown_open=$(echo "$open_users" \
                | sort | uniq -c | sort -nr \
                | awk '{printf "%s(%s) ", $2, $1}')
        fi
    fi

    if [ -n "$breakdown_compute" ] || [ -n "$breakdown_open" ]; then
        line="$TIMESTAMP 🎯 GPU $gpu_id: compute{ $breakdown_compute} process{ $breakdown_open}"
        echo "$line" | tee -a "$TMP_LOG"
    fi

    if [ "$COUNT" -gt "$GPU_THRESHOLD" ]; then
        msg="⚠️ GPU $gpu_id has $COUNT active processes! Exceeds threshold $GPU_THRESHOLD."
        echo "$TIMESTAMP $msg" | tee -a "$TMP_LOG"
        wall <<< "$msg"
    fi
done


# ========== CPU/MEM check ==========
MEM_LIMIT_KB=$((500 * 1024 * 1024))
total_cores=$(nproc)
cpu_threshold=$(echo "$total_cores * 100 * 0.7" | bc)

# 所有 sudo 用户
sudo_users=$(getent group sudo | cut -d: -f4 | tr ',' ' ')

# 所有 UID ≥ 1000 用户（包括 sudo 用户）
users=$(getent passwd {1000..60000} | cut -d: -f1)

for user in $users; do
    [[ "$user" =~ ^nixbld ]] && continue

    cpu_usage=$(ps -u "$user" -o %cpu= | awk '{sum+=$1} END {print sum + 0}')
    mem_usage_kb=$(ps -u "$user" -o rss= | awk '{sum+=$1} END {print sum + 0}')
    mem_usage_gb=$((mem_usage_kb / 1024 / 1024))

    if (( $(echo "$cpu_usage > 20" | bc -l) )) || [ "$mem_usage_kb" -gt $((20 * 1024 * 1024)) ]; then
        line="$TIMESTAMP [$user] CPU=${cpu_usage}% MEM=${mem_usage_gb}GB"
        echo "$line" | tee -a "$TMP_LOG"
    fi

    # 仅对非 sudo 用户广播警告
    if [[ ! " ${sudo_users[@]} " =~ " $user " ]]; then
        exceed_cpu=$(echo "$cpu_usage > $cpu_threshold" | bc | awk '{printf "%d", $1}')
        exceed_mem=0
        if [ "$mem_usage_kb" -gt "$MEM_LIMIT_KB" ]; then exceed_mem=1; fi

        if [ "$exceed_cpu" -eq 1 ] || [ "$exceed_mem" -eq 1 ]; then
            warning="⚠️  User '$user' is using high resources: CPU=${cpu_usage}% MEM=${mem_usage_gb}GB"
            for tty in $(who | awk -v u="$user" '$1 == u {print $2}'); do
                echo "$warning" | write "$user" "$tty"
            done
            for sudo_user in $sudo_users; do
                for tty in $(who | awk -v u="$sudo_user" '$1 == u {print $2}'); do
                    echo "$warning" | write "$sudo_user" "$tty"
                done
            done
        fi
    fi
done

# ========== prepend log (monthly file) ==========
YEAR_MONTH=$(date '+%Y-%m')
LOG_PATH="/data/public/system-tools/system_${YEAR_MONTH}.log"

if [ -s "$TMP_LOG" ]; then
    touch "$LOG_PATH"
    cat "$TMP_LOG" "$LOG_PATH" > "${LOG_PATH}.tmp" && mv "${LOG_PATH}.tmp" "$LOG_PATH"
fi

rm -f "$TMP_LOG"
