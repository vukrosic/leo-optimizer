#!/usr/bin/env python3
"""
GPU Monitor - Console-based GPU monitoring tool
Logs GPU memory usage and utilization periodically
"""

import time
import sys
import argparse
from datetime import datetime

try:
    import pynvml
    pynvml.nvmlInit()
except ImportError:
    print("Error: pynvml not installed. Install with: pip install pynvml")
    print("Note: Requires NVIDIA GPU and NVIDIA drivers")
    sys.exit(1)

def get_gpu_info(device_id=0):
    """Get GPU information for the specified device"""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        name = pynvml.nvmlDeviceGetName(handle)

        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_mb = mem_info.used // 1024 // 1024
        mem_total_mb = mem_info.total // 1024 // 1024
        mem_percent = (mem_info.used / mem_info.total) * 100

        # Utilization info
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util_info.gpu
        mem_util = util_info.memory

        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        return {
            'name': name,
            'memory_used_mb': mem_used_mb,
            'memory_total_mb': mem_total_mb,
            'memory_percent': mem_percent,
            'gpu_utilization': gpu_util,
            'memory_utilization': mem_util,
            'temperature': temp
        }
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def format_gpu_info(info):
    """Format GPU information for display"""
    if not info:
        return "Unable to get GPU information"

    return (f"GPU: {info['name']} | "
            f"Mem: {info['memory_used_mb']}/{info['memory_total_mb']} MB "
            f"({info['memory_percent']:.1f}%) | "
            f"GPU Util: {info['gpu_utilization']}% | "
            f"Mem Util: {info['memory_utilization']}% | "
            f"Temp: {info['temperature']}°C")

def main():
    parser = argparse.ArgumentParser(description='GPU Monitor - Monitor GPU usage periodically')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Logging interval in seconds (default: 2.0)')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--log-file', type=str,
                       help='Optional log file to save output')

    args = parser.parse_args()

    # Check if device exists
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if args.device >= device_count:
            print(f"Error: GPU device {args.device} not found. Available devices: 0-{device_count-1}")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing GPU monitoring: {e}")
        sys.exit(1)

    print(f"Starting GPU monitor for device {args.device}")
    print(f"Logging every {args.interval} seconds")
    if args.log_file:
        print(f"Saving log to: {args.log_file}")
    print("Press Ctrl+C to stop\n")

    log_file = open(args.log_file, 'a') if args.log_file else None

    try:
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            gpu_info = get_gpu_info(args.device)

            if gpu_info:
                log_line = f"[{timestamp}] {format_gpu_info(gpu_info)}"
                print(log_line)

                if log_file:
                    log_file.write(log_line + '\n')
                    log_file.flush()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopping GPU monitor...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if log_file:
            log_file.close()
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
