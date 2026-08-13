# Pre-resize correction snapshot

The first C-v2 run generated an explicit 65-share FTNT order from the signal
close. At the 2026-07-30 open, only 59 shares were affordable, so the old paper
path skipped the whole order. These files preserve that erroneous pre-fix
state. The active C account was reset to 2026-07-29 and replayed with
execution-price quantity capping.
