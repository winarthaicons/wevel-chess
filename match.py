#!/usr/bin/env python3
"""
Systematic UCI match: wevelchess_x64.exe vs stockfish_baseline_x64.exe
Runs N games with fixed time control and reports results + UCI parameter comparison.
"""

import subprocess
import threading
import time
import sys
import os
import re

WINE = "wine64"
WINEPREFIX = "/home/ubuntu/.wine"
WINEDEBUG = "-all"

WIN_DIR = "/home/ubuntu/stockfish_project/Windows"
ENGINE1_EXE = os.path.join(WIN_DIR, "wevelchess_x64.exe")
ENGINE2_EXE = os.path.join(WIN_DIR, "stockfish_baseline_x64.exe")

ENGINE1_NAME = "Wevel-x64"
ENGINE2_NAME = "SF11-baseline-x64"

# Time control: movetime in milliseconds per move
MOVETIME_MS = 100   # 100ms per move, fast but meaningful
NUM_GAMES   = 20    # 10 game pairs (each colour)

# Opening positions (FEN list) — diverse set to reduce opening bias
OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # start
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",  # Italian
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # open game
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",   # Scandinavian
    "rnbqkb1r/pppp1ppp/4pn2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",   # French Tarrasch
    "r1bqkbnr/pppp1ppp/2n5/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4",  # Scotch
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3",  # Caro-Kann
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # Sicilian
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", # open game
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",   # king's pawn
]

ENV = {"WINEDEBUG": WINEDEBUG, "WINEPREFIX": WINEPREFIX, "HOME": "/home/ubuntu"}

def start_engine(exe_path):
    cmd = [WINE, exe_path]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, **ENV},
        cwd=WIN_DIR,
    )
    return proc

def send(proc, msg):
    proc.stdin.write((msg + "\n").encode())
    proc.stdin.flush()

def read_until(proc, keyword, timeout=10.0):
    lines = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        proc.stdout.flush()
        line = proc.stdout.readline().decode(errors="replace").rstrip()
        if line:
            lines.append(line)
        if keyword in line:
            break
    return lines

def get_uci_info(proc):
    send(proc, "uci")
    return read_until(proc, "uciok", timeout=10)

def init_engine(proc):
    send(proc, "ucinewgame")
    send(proc, "isready")
    read_until(proc, "readyok", timeout=10)

def get_best_move(proc, fen, moves_so_far, movetime_ms):
    pos_cmd = f"position fen {fen}"
    if moves_so_far:
        pos_cmd += " moves " + " ".join(moves_so_far)
    send(proc, pos_cmd)
    send(proc, f"go movetime {movetime_ms}")
    lines = read_until(proc, "bestmove", timeout=movetime_ms / 1000 + 5.0)
    for line in reversed(lines):
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] != "(none)":
                return parts[1]
    return None

def play_game(e1_proc, e2_proc, opening_fen, e1_is_white, movetime_ms):
    """
    Play one game. Returns:
      1  = white wins
      0  = draw
      -1 = black wins
      Also returns move count and termination reason.
    """
    moves = []
    max_moves = 200  # cap to avoid infinite games

    # Map colour to proc
    white_proc = e1_proc if e1_is_white else e2_proc
    black_proc = e2_proc if e1_is_white else e1_proc

    for move_num in range(max_moves):
        side_to_move = "white" if move_num % 2 == 0 else "black"
        current_proc = white_proc if side_to_move == "white" else black_proc

        bm = get_best_move(current_proc, opening_fen, moves, movetime_ms)
        if bm is None:
            # Engine returned no move — treat as loss for that side
            return (1 if side_to_move == "black" else -1), move_num, "no_move"
        moves.append(bm)

        # Simple draw by move count
        if move_num >= max_moves - 1:
            return 0, move_num, "max_moves"

    return 0, max_moves, "max_moves"

def format_uci_diff(e1_lines, e2_lines):
    """Compare UCI option lines between the two engines."""
    e1_opts = {l for l in e1_lines if l.startswith("option")}
    e2_opts = {l for l in e2_lines if l.startswith("option")}
    only_e1 = sorted(e1_opts - e2_opts)
    only_e2 = sorted(e2_opts - e1_opts)
    return only_e1, only_e2

def main():
    print("=" * 60)
    print("  WEVELCHESS x64 vs STOCKFISH BASELINE x64")
    print("  Systematic UCI Match Report")
    print("=" * 60)

    # --- Start engines ---
    print("\n[1] Starting engines...")
    e1 = start_engine(ENGINE1_EXE)
    e2 = start_engine(ENGINE2_EXE)
    time.sleep(1.5)

    # --- UCI Parameter Verification ---
    print("\n[2] UCI Parameter Verification")
    print("-" * 40)
    e1_uci = get_uci_info(e1)
    e2_uci = get_uci_info(e2)

    print(f"\n  {ENGINE1_NAME} UCI identity:")
    for l in e1_uci[:3]:
        if l.startswith("id"):
            print(f"    {l}")

    print(f"\n  {ENGINE2_NAME} UCI identity:")
    for l in e2_uci[:3]:
        if l.startswith("id"):
            print(f"    {l}")

    only_e1, only_e2 = format_uci_diff(e1_uci, e2_uci)
    print(f"\n  Options in {ENGINE1_NAME} only (Wevel additions):")
    for o in only_e1:
        print(f"    + {o}")
    print(f"\n  Options in {ENGINE2_NAME} only:")
    for o in only_e2:
        print(f"    - {o}")
    if not only_e1 and not only_e2:
        print("    (identical)")

    # Check all standard SF11 options are present in Wevel
    sf11_opts = {l.split("name ")[1].split(" type")[0] for l in e2_uci if "option name" in l}
    wevel_opts = {l.split("name ")[1].split(" type")[0] for l in e1_uci if "option name" in l}
    missing = sf11_opts - wevel_opts
    print(f"\n  Standard SF11 options missing from Wevel: ", end="")
    print("NONE (all present)" if not missing else ", ".join(sorted(missing)))
    print(f"  Total Wevel UCI options: {len(wevel_opts)}")
    print(f"  Total SF11 UCI options:  {len(sf11_opts)}")

    # --- Engine initialization ---
    print("\n[3] Initializing engines for match...")
    init_engine(e1)
    init_engine(e2)

    # --- Play games ---
    print(f"\n[4] Running {NUM_GAMES} games ({NUM_GAMES//2} pairs, {MOVETIME_MS}ms/move)")
    print("-" * 40)

    e1_wins = 0; e2_wins = 0; draws = 0
    game_log = []

    opening_cycle = OPENINGS * ((NUM_GAMES // 2 // len(OPENINGS)) + 1)

    for pair in range(NUM_GAMES // 2):
        opening = opening_cycle[pair]
        short_fen = opening[:30] + "..."

        # Game A: e1=White, e2=Black
        result_a, moves_a, reason_a = play_game(e1, e2, opening, True, MOVETIME_MS)
        # result_a: 1=white(e1) wins, -1=black(e2) wins
        if result_a == 1:   e1_wins += 1; r_a = f"{ENGINE1_NAME} wins (W)"
        elif result_a == -1: e2_wins += 1; r_a = f"{ENGINE2_NAME} wins (B)"
        else:               draws += 1;   r_a = "Draw"

        # Game B: e1=Black, e2=White  (colours swapped)
        result_b, moves_b, reason_b = play_game(e1, e2, opening, False, MOVETIME_MS)
        # result_b from white's perspective: 1=white(e2) wins, -1=black(e1) wins
        if result_b == 1:    e2_wins += 1; r_b = f"{ENGINE2_NAME} wins (W)"
        elif result_b == -1: e1_wins += 1; r_b = f"{ENGINE1_NAME} wins (B)"
        else:                draws += 1;   r_b = "Draw"

        print(f"  Pair {pair+1:2d}: [{short_fen}]")
        print(f"           A: {r_a} ({moves_a} moves)  B: {r_b} ({moves_b} moves)")
        game_log.append((opening, r_a, r_b, moves_a, moves_b))

    # --- Final Results ---
    total = e1_wins + e2_wins + draws
    e1_pts = e1_wins + 0.5 * draws
    e2_pts = e2_wins + 0.5 * draws

    print("\n" + "=" * 60)
    print("  MATCH RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {ENGINE1_NAME:30s}: {e1_wins:3d}W / {draws:3d}D / {e2_wins:3d}L  = {e1_pts:5.1f} pts")
    print(f"  {ENGINE2_NAME:30s}: {e2_wins:3d}W / {draws:3d}D / {e1_wins:3d}L  = {e2_pts:5.1f} pts")
    print(f"  Total games: {total}")
    if e1_pts > e2_pts:
        margin = e1_pts - e2_pts
        print(f"\n  VERDICT: {ENGINE1_NAME} leads by {margin:.1f} points")
    elif e2_pts > e1_pts:
        margin = e2_pts - e1_pts
        print(f"\n  VERDICT: {ENGINE2_NAME} leads by {margin:.1f} points")
    else:
        print(f"\n  VERDICT: Match tied at {e1_pts:.1f} points each")
    print("=" * 60)

    # --- Cleanup ---
    for proc in [e1, e2]:
        try:
            send(proc, "quit")
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    print("\n  Match complete. Engines shut down.")

if __name__ == "__main__":
    os.chdir(WIN_DIR)
    main()
