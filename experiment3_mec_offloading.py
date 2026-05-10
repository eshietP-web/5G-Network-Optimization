"""
Experiment 3: Mobile Edge Computing (MEC) Task Offloading
=========================================================
Compares three task offloading strategies:
  1. Greedy Threshold-Based                 - Conventional baseline
  2. Lyapunov Optimization                  - Mathematical optimization
  3. DRL-based Offloading (PPO)             - Deep Reinforcement Learning

Metrics: Average Task Completion Latency (ms)
System:  1 MEC Server co-located with Macro BS
         10^10 cycles/s computation capacity
         Mobile users with heterogeneous task sizes (10^6 to 10^8 CPU cycles)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────
SEEDS      = [42, 7, 13, 21, 99]
N_EPISODES = 600

# ── System Parameters ────────────────────────────────────────
N_USERS       = 50          # Normal load
N_USERS_HIGH  = 100         # High load
F_MEC         = 1e10        # MEC computation capacity (cycles/s)
F_LOCAL       = 1e8         # Local device computation (cycles/s)
BW_UL         = 10e6        # Uplink bandwidth per user (Hz)
P_TX          = 0.1         # Transmit power (W)
N0            = 1e-13       # Noise power spectral density

# Task parameters
TASK_MIN      = 1e6         # Min task size (CPU cycles)
TASK_MAX      = 1e8         # Max task size (CPU cycles)
DATA_MIN      = 0.1e6       # Min data size (bits)
DATA_MAX      = 2e6         # Max data size (bits)
SNR_THRESHOLD = 10.0        # SNR threshold for greedy offloading (dB)

# ═══════════════════════════════════════════════════════════════
# Channel and Task Generation
# ═══════════════════════════════════════════════════════════════
def gen_users(n_users, seed):
    """Generate user task and channel parameters"""
    rng     = np.random.RandomState(seed)
    tasks   = rng.uniform(TASK_MIN, TASK_MAX, n_users)
    data    = rng.uniform(DATA_MIN, DATA_MAX, n_users)
    snr_db  = rng.normal(15, 6, n_users)
    snr_lin = 10**(snr_db / 10.0)
    ul_rate = BW_UL * np.log2(1 + snr_lin)
    return tasks, data, snr_db, snr_lin, ul_rate

def compute_latency(tasks, data, ul_rate, offload_mask):
    """
    Compute per-user task completion latency.
    offload_mask: boolean array, True = offload to MEC
    Returns average latency in ms.
    """
    n         = len(tasks)
    latency   = np.zeros(n)
    n_offload = max(offload_mask.sum(), 1)

    for i in range(n):
        if offload_mask[i]:
            t_tx      = data[i] / (ul_rate[i] + 1e-6)
            f_share   = F_MEC / n_offload
            t_mec     = tasks[i] / (f_share + 1e-6)
            latency[i] = (t_tx + t_mec) * 1000
        else:
            t_local   = tasks[i] / F_LOCAL
            latency[i] = t_local * 1000

    return latency

# ═══════════════════════════════════════════════════════════════
# Method 1: Greedy Threshold-Based Offloading
# ═══════════════════════════════════════════════════════════════
def greedy_offload(tasks, data, snr_db, snr_lin, ul_rate):
    """Offload if SNR > threshold AND task is large enough."""
    offload = (snr_db >= SNR_THRESHOLD) & (tasks >= TASK_MIN * 10)
    lat     = compute_latency(tasks, data, ul_rate, offload)
    return lat, offload

# ═══════════════════════════════════════════════════════════════
# Method 2: Lyapunov Optimization
# ═══════════════════════════════════════════════════════════════
def lyapunov_offload(tasks, data, snr_db, snr_lin, ul_rate, V=1e6):
    """Lyapunov drift-plus-penalty optimization."""
    n       = len(tasks)
    queue   = np.zeros(n)
    offload = np.zeros(n, dtype=bool)

    for i in range(n):
        lat_local   = tasks[i] / F_LOCAL * 1000
        n_est       = max(n // 2, 1)
        f_share     = F_MEC / n_est
        lat_offload = (data[i] / (ul_rate[i]+1e-6) +
                       tasks[i] / (f_share+1e-6)) * 1000

        penalty_local   = V * lat_local   + queue[i] * lat_local
        penalty_offload = V * lat_offload + queue[i] * lat_offload
        offload[i]      = penalty_offload < penalty_local

        arrival  = tasks[i] / TASK_MAX
        service  = 1.0 if offload[i] else 0.5
        queue[i] = max(queue[i] + arrival - service, 0)

    lat = compute_latency(tasks, data, ul_rate, offload)
    return lat, offload

# ═══════════════════════════════════════════════════════════════
# Method 3: DRL-PPO Offloading
# ═══════════════════════════════════════════════════════════════
def drl_ppo_offload(tasks, data, snr_db, snr_lin, ul_rate,
                    n_episodes=N_EPISODES, seed=42):
    """PPO-style policy gradient for MEC offloading."""
    rng    = np.random.RandomState(seed)
    n      = len(tasks)

    # Warm start from Lyapunov
    _, best_offload = lyapunov_offload(tasks, data, snr_db, snr_lin, ul_rate)
    best_lat        = compute_latency(tasks, data, ul_rate, best_offload)
    best_avg_lat    = best_lat.mean()

    logits = np.where(best_offload, 1.0, -1.0).astype(float)
    lr     = 0.15
    clip   = 0.2

    for ep in range(n_episodes):
        probs     = 1 / (1 + np.exp(-logits))
        offload_t = rng.rand(n) < probs
        lat_t     = compute_latency(tasks, data, ul_rate, offload_t)
        avg_lat_t = lat_t.mean()

        advantage = -(avg_lat_t - best_avg_lat)
        grad      = np.where(offload_t,
                             advantage * (1 - probs),
                             -advantage * probs)
        logits   += lr * grad
        logits    = np.clip(logits, -5, 5)

        if avg_lat_t < best_avg_lat:
            best_avg_lat  = avg_lat_t
            best_offload  = offload_t.copy()

        lr = max(lr * 0.997, 0.01)

    final_lat = compute_latency(tasks, data, ul_rate, best_offload)
    return final_lat, best_offload

# ═══════════════════════════════════════════════════════════════
# Run Experiment
# ═══════════════════════════════════════════════════════════════
print("="*60)
print("Experiment 3: MEC Task Offloading Optimization")
print("="*60)
print(f"Normal load: {N_USERS} UEs, High load: {N_USERS_HIGH} UEs")
print(f"MEC capacity: {F_MEC:.0e} cycles/s")
print(f"Running over {len(SEEDS)} seeds...\n")

results = {
    'normal': {'greedy': [], 'lyapunov': [], 'drl': []},
    'high':   {'greedy': [], 'lyapunov': [], 'drl': []}
}

for s, seed in enumerate(SEEDS):
    print(f"  Seed {seed} ({s+1}/{len(SEEDS)})...")
    for load_name, n_ue in [('normal', N_USERS), ('high', N_USERS_HIGH)]:
        tasks, data, snr_db, snr_lin, ul_rate = gen_users(n_ue, seed)

        lat_g, _ = greedy_offload(tasks, data, snr_db, snr_lin, ul_rate)
        lat_l, _ = lyapunov_offload(tasks, data, snr_db, snr_lin, ul_rate)
        lat_d, _ = drl_ppo_offload(tasks, data, snr_db, snr_lin, ul_rate,
                                    seed=seed)

        results[load_name]['greedy'].append(lat_g.mean())
        results[load_name]['lyapunov'].append(lat_l.mean())
        results[load_name]['drl'].append(lat_d.mean())

avg = {}
for load in ['normal', 'high']:
    avg[load] = {}
    for method in ['greedy', 'lyapunov', 'drl']:
        avg[load][method] = np.mean(results[load][method])

print("\n" + "="*60)
print("KEY RESULTS — Average Task Completion Latency (ms):")
print(f"\n  Normal Load ({N_USERS} UEs/km²):")
print(f"    Greedy     : {avg['normal']['greedy']:.1f} ms")
print(f"    Lyapunov   : {avg['normal']['lyapunov']:.1f} ms")
print(f"    DRL-PPO    : {avg['normal']['drl']:.1f} ms")

print(f"\n  High Load ({N_USERS_HIGH} UEs/km²):")
print(f"    Greedy     : {avg['high']['greedy']:.1f} ms")
print(f"    Lyapunov   : {avg['high']['lyapunov']:.1f} ms")
print(f"    DRL-PPO    : {avg['high']['drl']:.1f} ms")

reduction = (avg['normal']['greedy'] - avg['normal']['drl']) / \
             avg['normal']['greedy'] * 100
print(f"\n  DRL Latency Reduction over Greedy (normal load): {reduction:.1f}%")
print("="*60)

# ── Plot ─────────────────────────────────────────────────────
methods     = ['Greedy\n(Threshold)', 'Lyapunov\nOptimization', 'DRL-PPO']
normal_vals = [avg['normal']['greedy'],
               avg['normal']['lyapunov'],
               avg['normal']['drl']]
high_vals   = [avg['high']['greedy'],
               avg['high']['lyapunov'],
               avg['high']['drl']]

x     = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - width/2, normal_vals, width,
               label=f'Normal Load ({N_USERS} UEs/km²)',
               color='steelblue', edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x + width/2, high_vals, width,
               label=f'High Load ({N_USERS_HIGH} UEs/km²)',
               color='coral', edgecolor='black', linewidth=0.7)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Average Task Completion Latency (ms)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig3_mec_results.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as: fig3_mec_results.png")
