## Wevel 1.0

Wevel is a UCI chess engine derived from Stockfish 11.
Author: Fatih W

Stockfish is a free UCI chess engine derived from Glaurung 2.1 by Tord Romstad,
Marco Costalba, Joona Kiiski, and Gary Linscott. Wevel modifies Stockfish 11
and is distributed under the same GPL v3 license.

Requires a UCI-compatible GUI (Arena, Cute Chess, Fritz, Scid, etc.).


## Files

  * Readme.md — this file.
  * Copying.txt — GNU General Public License version 3.
  * src — full source code with Makefile for Unix-like systems.


## UCI parameters

  * Debug Log File — log all engine communication.
  * Contempt — positive values favor middlegame positions, avoid draws.
  * Analysis Contempt — set contempt side or disable.
  * Threads — CPU threads for searching.
  * Hash — hash table size in MB.
  * Clear Hash — clear the hash table.
  * Ponder — ponder while opponent thinks.
  * MultiPV — output N best lines.
  * Skill Level — lower for weaker play.
  * UCI_LimitStrength — aim for target Elo rating.
  * UCI_Elo — target Elo (requires UCI_LimitStrength).
  * Move Overhead — time delay compensation in ms.
  * Minimum Thinking Time — minimum search time per move in ms.
  * Slow Mover — lower = less time, higher = more time.
  * nodestime — use nodes searched instead of wall time.
  * UCI_Chess960 — enable Chess960 mode.
  * UCI_AnalyseMode — analysis mode toggle.
  * SyzygyPath — path to Syzygy tablebase files.
  * SyzygyProbeDepth — minimum depth for TB probing.
  * Syzygy50MoveRule — 50-move rule in TB probes.
  * SyzygyProbeLimit — max pieces for TB probing.


## Compiling

On Unix-like systems, compile with the included Makefile. Run `make help`
for targets. For MSVC, set switches per *types.h*.


## Terms of use

Wevel is free, distributed under **GPL v3**. You may distribute, modify, and
sell it, but must always include the full source code (or a pointer to it).
All changes must also be under GPL. See *Copying.txt*.