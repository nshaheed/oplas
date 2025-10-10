import psutil
import sys
import time
import datetime


def measure_peaks(pid: int):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"No such process with PID {pid}")
        return

    start_time = time.time()
    max_mem = 0
    total_procs = set()
    max_active_procs = 0
    max_cpu_percents = []

    try:
        while True:
            with proc.oneshot():
                try:
                    mem = proc.memory_info().rss  # in bytes
                    max_mem = max(max_mem, mem)

                    children = proc.children(recursive=True)
                    all_procs = [proc] + children
                    total_procs.update(p.pid for p in all_procs)

                    # Count active processes (those using CPU time)
                    active = 0
                    percents = []
                    for p in all_procs:
                        try:
                            # cpu_percent = p.cpu_percent(interval=0.5)  # short sample
                            cpu_percent = p.cpu_percent()  # short sample
                            print(f"{p.pid=}, {cpu_percent=}")
                            if cpu_percent > 0:
                                active += 1
                            percents.append(cpu_percent)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

                    max_active_procs = max(max_active_procs, active)
                    if percents:
                        max_cpu_percents = percents

                except psutil.NoSuchProcess:
                    break

            time.sleep(1)

    except KeyboardInterrupt:
        pass

    # Compute duration
    duration = int(time.time() - start_time)
    duration_str = str(datetime.timedelta(seconds=duration))

    # Print summary
    print(f"Time:           {duration_str}")
    print(f"Memory:         {max_mem / (1024**3):.1f} GB")
    print(f"Cores:          {psutil.cpu_count(logical=True)}")
    print(f"Total_procs:    {len(total_procs)}")
    print(f"Active_procs:   {max_active_procs}")
    if max_cpu_percents:
        print("Proc(%):", "  ".join(f"{p:.1f}" for p in max_cpu_percents))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <pid>")
        sys.exit(1)

    measure_peaks(int(sys.argv[1]))
