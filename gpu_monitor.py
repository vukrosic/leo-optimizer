#!/usr/bin/env python3
"""
GPU Monitor - Console-based GPU monitoring tool
Logs GPU memory usage and utilization periodically with colored indicators
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

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'

# Global variable to track color preference
NO_COLOR = False

def get_usage_indicator(percentage):
    """Get colored emoji indicator based on usage percentage"""
    if NO_COLOR:
        if percentage < 30:
            return "🟢"
        elif percentage < 70:
            return "🟡"
        else:
            return "🔴"

    if percentage < 30:
        return f"{Colors.GREEN}🟢{Colors.RESET}"
    elif percentage < 70:
        return f"{Colors.YELLOW}🟡{Colors.RESET}"
    else:
        return f"{Colors.RED}🔴{Colors.RESET}"

def get_usage_color(percentage):
    """Get color code based on usage percentage"""
    if NO_COLOR:
        return ""
    if percentage < 30:
        return Colors.GREEN
    elif percentage < 70:
        return Colors.YELLOW
    else:
        return Colors.RED

def colorize_percentage(percentage):
    """Colorize percentage text based on usage level"""
    if NO_COLOR:
        return f"{percentage:.1f}%"
    color = get_usage_color(percentage)
    return f"{color}{percentage:.1f}%{Colors.RESET}"

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
    """Format GPU information for display with colored indicators"""
    if not info:
        return "Unable to get GPU information"

    mem_indicator = get_usage_indicator(info['memory_percent'])
    gpu_indicator = get_usage_indicator(info['gpu_utilization'])
    mem_util_indicator = get_usage_indicator(info['memory_utilization'])

    mem_percent_colored = colorize_percentage(info['memory_percent'])
    gpu_util_colored = colorize_percentage(info['gpu_utilization'])
    mem_util_colored = colorize_percentage(info['memory_utilization'])

    if NO_COLOR:
        return (f"GPU: {info['name']} | "
                f"{mem_indicator} Mem: {info['memory_used_mb']}/{info['memory_total_mb']} MB "
                f"({mem_percent_colored}) | "
                f"{gpu_indicator} GPU Util: {gpu_util_colored} | "
                f"{mem_util_indicator} Mem Util: {mem_util_colored} | "
                f"Temp: {info['temperature']}°C")

    return (f"{Colors.BOLD}GPU:{Colors.RESET} {Colors.CYAN}{info['name']}{Colors.RESET} | "
            f"{mem_indicator} {Colors.BOLD}Mem:{Colors.RESET} {info['memory_used_mb']}/{info['memory_total_mb']} MB "
            f"({mem_percent_colored}) | "
            f"{gpu_indicator} {Colors.BOLD}GPU Util:{Colors.RESET} {gpu_util_colored} | "
            f"{mem_util_indicator} {Colors.BOLD}Mem Util:{Colors.RESET} {mem_util_colored} | "
            f"{Colors.BOLD}Temp:{Colors.RESET} {Colors.MAGENTA}{info['temperature']}°C{Colors.RESET}")

def strip_ansi_codes(text):
    """Remove ANSI color codes from text for log files"""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def main():
    parser = argparse.ArgumentParser(description='GPU Monitor - Monitor GPU usage periodically with colored indicators')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Logging interval in seconds (default: 2.0)')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--log-file', type=str,
                       help='Optional log file to save output')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')

    args = parser.parse_args()

    # Set global color preference
    global NO_COLOR
    NO_COLOR = args.no_color

    # Check if device exists
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if args.device >= device_count:
            print(f"Error: GPU device {args.device} not found. Available devices: 0-{device_count-1}")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing GPU monitoring: {e}")
        sys.exit(1)

    if not NO_COLOR:
        print(f"{Colors.BOLD}{Colors.BLUE}🚀 Starting GPU monitor for device {args.device}{Colors.RESET}")
        print(f"📊 Logging every {args.interval} seconds")
        if args.log_file:
            print(f"💾 Saving log to: {args.log_file}")
        print(f"🎨 Color indicators: {Colors.GREEN}🟢{Colors.RESET} Low (<30%) | {Colors.YELLOW}🟡{Colors.RESET} Medium (30-70%) | {Colors.RED}🔴{Colors.RESET} High (>70%)")
        print(f"{Colors.BOLD}Press Ctrl+C to stop{Colors.RESET}\n")
    else:
        print(f"🚀 Starting GPU monitor for device {args.device}")
        print(f"📊 Logging every {args.interval} seconds")
        if args.log_file:
            print(f"💾 Saving log to: {args.log_file}")
        print(f"🎨 Color indicators: 🟢 Low (<30%) | 🟡 Medium (30-70%) | 🔴 High (>70%)")
        print(f"Press Ctrl+C to stop\n")

    log_file = open(args.log_file, 'a') if args.log_file else None

    try:
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            gpu_info = get_gpu_info(args.device)

            if gpu_info:
                colored_line = f"[{timestamp}] {format_gpu_info(gpu_info)}"
                print(colored_line)

                if log_file:
                    # Strip ANSI codes for log file
                    clean_line = f"[{timestamp}] {strip_ansi_codes(format_gpu_info(gpu_info))}"
                    log_file.write(clean_line + '\n')
                    log_file.flush()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        if not NO_COLOR:
            print(f"\n{Colors.BOLD}{Colors.BLUE}🛑 Stopping GPU monitor...{Colors.RESET}")
        else:
            print("\n🛑 Stopping GPU monitor...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if log_file:
            log_file.close()
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
