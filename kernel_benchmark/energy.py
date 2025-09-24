#!/usr/bin/env python3
import time
import subprocess
import statistics
from jtop import jtop

# —— 配置区 ——
INTERVAL = 1  # 采样间隔 (秒).最小值为1
BASELINE_SEC = 10  # 基线采样时长 (秒)
COOLDOWN_SEC = 5  # 每组测试之间的冷却时间 (秒)
POWER_KEY = "Power VIN_SYS_5V0"  # tegrastats/ jtop 中的 5V 输入功率 字段 (mW)

# 定义四组测试命令
KERNEL_CMDS = [
    {
        "name": "Test 1 (M=27648, K=9216, N=16)",
        "cmd": ["./kernel_utilization", "27648", "9216", "16", "90", "5"],
    },
    {
        "name": "Test 2 (M=9216, K=9216, N=16)",
        "cmd": ["./kernel_utilization", "9216", "9216", "16", "90", "5"],
    },
    {
        "name": "Test 3 (M=36864, K=9216, N=16)",
        "cmd": ["./kernel_utilization", "36864", "9216", "16", "90", "5"],
    },
    {
        "name": "Test 4 (M=9216, K=36864, N=16)",
        "cmd": ["./kernel_utilization", "9216", "36864", "16", "90", "5"],
    },
]
# ————————


def sample_baseline(duration: float, interval: float):
    samples = []
    with jtop() as jet:
        # print("Available jtop.stats keys:", list(jet.stats.keys()))
        # return
        t_end = time.time() + duration
        while time.time() < t_end and jet.ok():
            samples.append(jet.stats[POWER_KEY])
            time.sleep(interval)
    return samples


def measure_run(interval: float, power_key: str, cmd: list):
    ts, ps = [], []
    with jtop() as jet:
        time.sleep(0.1)
        proc = subprocess.Popen(cmd)
        while proc.poll() is None and jet.ok():
            now = time.time()
            p = jet.stats[power_key]
            ts.append(now)
            ps.append(p)
            time.sleep(interval)
        # 完成后多读一次
        now = time.time()
        p = jet.stats[power_key]
        ts.append(now)
        ps.append(p)
    return ts, ps


def trapezoid_integral(times: list, powers: list, baseline: float):
    """
    针对 (times[i], powers[i]) 做梯形积分，powers 单位 mW，
    先减 baseline，结果返回 energy_mJ_s (mW·s)
    """
    E = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]  # s
        # 减去基线后的额外功率
        p0 = max(0.0, powers[i] - baseline)  # mW
        p1 = max(0.0, powers[i + 1] - baseline)
        E += (p0 + p1) / 2 * dt  # mW * s = mJ
    return E


def main():
    print("=" * 80)
    print("            GPU 内核能耗测试 - 多组测试")
    print("=" * 80)

    print(f"[*] 采集 Idle 基线功率 {BASELINE_SEC}s ...")
    base_samples = sample_baseline(BASELINE_SEC, INTERVAL)
    baseline = statistics.median(base_samples)  # 取中位数作为baseline
    print(f"[+] 基线功率 (median): {baseline:.1f} mW")
    print()

    results = []

    for i, test_config in enumerate(KERNEL_CMDS, 1):
        test_name = test_config["name"]
        test_cmd = test_config["cmd"]

        print(f"[*] 正在执行 {test_name}")
        print(f"    命令: {' '.join(test_cmd)}")

        # 运行测试
        ts, ps = measure_run(INTERVAL, POWER_KEY, test_cmd)

        # 计算结果
        t0, t1 = ts[0], ts[-1]
        duration = t1 - t0  # s
        energy_mJs = trapezoid_integral(ts, ps, baseline)
        energy_J = energy_mJs / 1000  # mW·s -> J
        energy_Wh = energy_J / 3600  # J -> Wh

        # 保存结果
        result = {
            "name": test_name,
            "duration": duration,
            "energy_J": energy_J,
            "energy_Wh": energy_Wh,
            "avg_power": (
                energy_mJs / duration if duration > 0 else 0
            ),  # 平均额外功率 mW
        }
        results.append(result)

        print(f"    持续时间: {duration:.3f} s")
        print(f"    消耗能量: {energy_J:.4f} J ({energy_Wh:.6f} Wh)")
        print(f"    平均额外功率: {result['avg_power']:.1f} mW")
        print()

        # 如果不是最后一个测试，等待冷却
        if i < len(KERNEL_CMDS):
            print(f"[*] 冷却等待 {COOLDOWN_SEC}s ...")
            time.sleep(COOLDOWN_SEC)
            print()

    # 输出总结报告
    print("=" * 85)
    print("                           测试总结报告")
    print("=" * 85)
    print(
        f"{'测试名称':<40} {'持续时间(s)':>10} {'能量(J)':>10} {'能量(Wh)':>12} {'平均功率(mW)':>12}"
    )
    print("-" * 85)

    total_energy_J = 0
    total_energy_Wh = 0

    for result in results:
        print(
            f"{result['name']:<40} {result['duration']:>10.3f} {result['energy_J']:>10.4f} "
            f"{result['energy_Wh']:>12.6f} {result['avg_power']:>12.1f}"
        )
        total_energy_J += result["energy_J"]
        total_energy_Wh += result["energy_Wh"]

    print("-" * 85)
    print(f"{'总计':<40} {'':<10} {total_energy_J:>10.4f} {total_energy_Wh:>12.6f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
