import os
import sys
import random
import argparse
from collections import deque, defaultdict

import numpy as np
import pandas as pd

from laplace_noise import noisy_position, noisy_speed, NOISE_SCALE

# ---- SUMO / TraCI setup -----------------------------------------------
if "SUMO_HOME" not in os.environ:
    import sumo
    os.environ["SUMO_HOME"] = os.path.dirname(sumo.__file__)
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402

SUMO_BINARY = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
NETWORK_DIR = os.path.join(os.path.dirname(__file__), "network")
SUMOCFG = os.path.join(NETWORK_DIR, "simulation.sumocfg")

COMM_RANGE = 150.0          # metres, V2V communication radius
NEIGHBOUR_TIMEOUT = 2.0     # seconds, matches Module 1 cleanup rule
HISTORY_LEN = 5             # last 5 speed readings, for DTW / avg_speed
PAIRS_PER_STEP = 6          # random pairs sampled per receiver per step
ATTACKER_FRACTION = 0.22    # share of real vehicles that behave as Sybil attackers
GHOST_JITTER_POS_STD = 4.0  # metres, how far a fake identity drifts from its parent
GHOST_JITTER_SPEED_STD = 1.5  # km/h equivalent jitter on ghost speed


def euclidean(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def dtw_distance(seq_a, seq_b):
    """Dynamic Time Warping distance, exactly per the slide's recurrence:
    Cost_ij = (x_i - y_j)^2 ; d_ij = Cost_ij + min(d_{i-1,j}, d_{i,j-1}, d_{i-1,j-1})
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0
    INF = float("inf")
    d = [[INF] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (seq_a[i - 1] - seq_b[j - 1]) ** 2
            d[i][j] = cost + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return float(d[n][m])


class VehicleTrack:
    """Rolling state for one identity (real vehicle OR sybil ghost)."""

    def __init__(self, vid, family_id, is_ghost=False):
        self.vid = vid
        self.family_id = family_id  # identities sharing a family_id are Sybil siblings
        self.is_ghost = is_ghost
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        self.first_seen = None
        self.last_seen = None
        self.speed_history = deque(maxlen=HISTORY_LEN)
        self.pos_window = deque(maxlen=HISTORY_LEN)  # for avg_speed over last 5 readings

    def update(self, t, x, y, speed):
        if self.first_seen is None:
            self.first_seen = t
        self.last_seen = t
        self.x, self.y, self.speed = x, y, speed
        self.speed_history.append(speed)

    @property
    def avg_speed(self):
        return float(np.mean(self.speed_history)) if self.speed_history else 0.0


def make_ghosts_for(track: VehicleTrack, rng: random.Random, np_rng: np.random.Generator):
    """Return 0-2 fabricated Sybil identities shadowing a real attacker vehicle."""
    n_ghosts = 0
    roll = rng.random()
    if roll < 0.55:
        n_ghosts = 1
    elif roll < 0.85:
        n_ghosts = 2
    ghosts = []
    for i in range(n_ghosts):
        gx = track.x + np_rng.normal(0, GHOST_JITTER_POS_STD)
        gy = track.y + np_rng.normal(0, GHOST_JITTER_POS_STD)
        gspeed = max(0.0, track.speed + np_rng.normal(0, GHOST_JITTER_SPEED_STD))
        ghosts.append((f"{track.vid}_sybil{i+1}", gx, gy, gspeed))
    return ghosts


def run(steps=600, out_csv="sumo_features_dataset.csv", seed=42, apply_privacy_noise=True):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    traci.start([SUMO_BINARY, "-c", SUMOCFG, "--no-warnings", "--no-step-log", "--seed", str(seed)])

    tracks = {}                       # vid -> VehicleTrack  (real + ghost)
    attacker_ids = set()              # real vehicle ids chosen as Sybil attackers
    rows = []

    t = 0
    while t < steps and (traci.simulation.getMinExpectedNumber() > 0 or t < 5):
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        real_ids = traci.vehicle.getIDList()

        # ---- decide which real vehicles are attackers (sticky decision) ----
        for vid in real_ids:
            if vid not in attacker_ids and vid not in tracks:
                if rng.random() < ATTACKER_FRACTION:
                    attacker_ids.add(vid)

        # ---- update real vehicle tracks ----
        active_this_step = {}
        for vid in real_ids:
            x, y = traci.vehicle.getPosition(vid)
            speed_kmh = traci.vehicle.getSpeed(vid) * 3.6
            if vid not in tracks:
                tracks[vid] = VehicleTrack(vid, family_id=vid, is_ghost=False)
            tr = tracks[vid]
            tr.update(sim_time, x, y, speed_kmh)
            active_this_step[vid] = tr

            # spawn/refresh ghost identities for attacker vehicles
            if vid in attacker_ids:
                for gid, gx, gy, gspeed in make_ghosts_for(tr, rng, np_rng):
                    if gid not in tracks:
                        tracks[gid] = VehicleTrack(gid, family_id=vid, is_ghost=True)
                    gtr = tracks[gid]
                    gtr.update(sim_time, gx, gy, gspeed)
                    active_this_step[gid] = gtr

        # ---- drop stale tracks (Module 1 periodic cleanup) ----
        stale = [vid for vid, tr in tracks.items()
                 if tr.last_seen is not None and sim_time - tr.last_seen > NEIGHBOUR_TIMEOUT]
        for vid in stale:
            del tracks[vid]

        active_ids = list(active_this_step.keys())
        if len(active_ids) < 3:
            t += 1
            continue

        # ---- per-sender density (vehicles within COMM_RANGE of the sender) ----
        density = {}
        positions = {vid: (active_this_step[vid].x, active_this_step[vid].y) for vid in active_ids}
        for vid in active_ids:
            cnt = 0
            for other in active_ids:
                if other == vid:
                    continue
                if euclidean(positions[vid], positions[other]) <= COMM_RANGE:
                    cnt += 1
            density[vid] = cnt

        # ---- pick a receiver (a real, non-ghost vehicle) ----
        real_active = [vid for vid in active_ids if not active_this_step[vid].is_ghost]
        if not real_active:
            t += 1
            continue
        receiver_id = rng.choice(real_active)
        receiver_pos = positions[receiver_id]

        # neighbours = anyone (other than receiver) within comm range
        neighbours = [vid for vid in active_ids
                      if vid != receiver_id and euclidean(receiver_pos, positions[vid]) <= COMM_RANGE]
        if len(neighbours) < 2:
            t += 1
            continue

        # ---- explicitly capture Sybil sibling pairs present in this neighbourhood ----
        by_family = defaultdict(list)
        for vid in neighbours:
            by_family[active_this_step[vid].family_id].append(vid)

        chosen = set()
        for fam_members in by_family.values():
            if len(fam_members) >= 2:
                for i in range(len(fam_members)):
                    for j in range(i + 1, len(fam_members)):
                        chosen.add(tuple(sorted((fam_members[i], fam_members[j]))))

        # ---- fill with cross-family Normal pairs to keep classes realistically balanced
        #      (target ~3 Normal pairs for every Sybil sibling pair, similar to a
        #      real deployment where most traffic is legitimate) ----
        n_sibling_pairs = len(chosen)
        target_normal = max(PAIRS_PER_STEP, n_sibling_pairs * 3)
        added_normal = 0
        attempts = 0
        while added_normal < target_normal and attempts < target_normal * 15:
            attempts += 1
            n_id, m_id = rng.sample(neighbours, 2)
            if active_this_step[n_id].family_id == active_this_step[m_id].family_id:
                continue  # would be a sibling pair, already covered above
            key = tuple(sorted((n_id, m_id)))
            if key in chosen:
                continue
            chosen.add(key)
            added_normal += 1

        for n_id, m_id in chosen:
            n_tr, m_tr = active_this_step[n_id], active_this_step[m_id]

            # simulate what the receiver actually hears: noisy broadcast (Module 5)
            if apply_privacy_noise:
                n_pos = noisy_position(n_tr.x, n_tr.y, rng=np_rng)
                m_pos = noisy_position(m_tr.x, m_tr.y, rng=np_rng)
                n_speed_hist = [noisy_speed(s, rng=np_rng) for s in n_tr.speed_history]
                m_speed_hist = [noisy_speed(s, rng=np_rng) for s in m_tr.speed_history]
            else:
                n_pos, m_pos = (n_tr.x, n_tr.y), (m_tr.x, m_tr.y)
                n_speed_hist, m_speed_hist = list(n_tr.speed_history), list(m_tr.speed_history)

            ed_n = euclidean(receiver_pos, n_pos)
            ed_m = euclidean(receiver_pos, m_pos)
            s_n = float(np.mean(n_speed_hist)) if n_speed_hist else 0.0
            s_m = float(np.mean(m_speed_hist)) if m_speed_hist else 0.0
            q_n = s_n * density.get(n_id, 0)
            q_m = s_m * density.get(m_id, 0)
            d_mn = dtw_distance(n_speed_hist, m_speed_hist)

            label = 1 if n_tr.family_id == m_tr.family_id else 0

            rows.append({
                "time": sim_time,
                "receiver_id": receiver_id,
                "vehicle_N": n_id,
                "vehicle_M": m_id,
                "ED_N": ed_n,
                "ED_M": ed_m,
                "S_N": s_n,
                "S_M": s_m,
                "q_N": q_n,
                "q_M": q_m,
                "D_MN": d_mn,
                "label": label,
            })

        t += 1

    traci.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    n_total = len(df)
    n_sybil = int(df["label"].sum()) if n_total else 0
    n_normal = n_total - n_sybil
    print("=" * 42)
    print("DATASET CREATED")
    print("=" * 42)
    print(f"Final Dataset:")
    print(f"  Total pairs: {n_total}")
    print(f"  Normal: {n_normal}")
    print(f"  Sybil: {n_sybil}")
    if n_total:
        print(f"  Sybil %: {100 * n_sybil / n_total:.1f}%")
    print(f"\nSaved to: {out_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--out", type=str, default="sumo_features_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-noise", action="store_true", help="disable Laplace privacy noise")
    args = parser.parse_args()
    run(steps=args.steps, out_csv=args.out, seed=args.seed, apply_privacy_noise=not args.no_noise)
