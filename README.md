# ML_Two_Agents_Transportation
This project aims to design and evaluate a two‑agent reinforcement learning system operating in the PD‑World environment.
theme credit to https://github.com/rdbende/Azure-ttk-theme

# Requirements
Install as requirements.txt
*Note: Tkinter (usually included with Python on Windows/macOS; on some Linux distros you may need to install python3-tk)

*Project structure:*
```
project_root/
  control_panel_gui.py
  run_experiment.py
  experiment/
    exp1.py
    exp2.py
    exp3.py
    exp4.py
  environment/
  agents/
  utils/
  visualization/
  artifacts/        # created automatically for outputs
```

# Method 1: Using the GUI (Recommended)

  *2. Launching using Tkinter GUI *

  From the project root: 
  ```
  python control_panel_gui.py
  ```
  A window should appear

  *3. GUI Layout*

The GUI is divided into 3 sections:

Configuration (top)

Run button (middle)

Live Output (large text area)

*3.1 Configuration*

Inside the Configuration frame you’ll see:

a) Experiment selection

A row of radio buttons:
```
1a, 1b, 1c, 2, 3, 4
```
These correspond to:
```
1a / 1b / 1c – Q-learning variants (different policies after warmup)

2 – SARSA (PEXPLOIT after warmup)

3 – Alpha sweep (re-runs Exp 1.c or Exp 2 with α in {0.15, 0.45})

4 – Pickup change mid-run (adapts to changed pickups after 3rd terminal)
```
Click one to select the experiment.

The GUI automatically enables/disables the Algorithm choice depending on the experiment:

For Exp 1: algorithm is fixed to Q-Learning.

For Exp 2: algorithm is fixed to SARSA.

For Exp 3 & 4: you can choose Q-Learning or SARSA.

b) Runs

Label: Runs:

Field: a small entry box (default 2)

This controls --runs in run_experiment.py and should match how many seeds you provide.

c) Algorithm

Radio buttons:

Q-Learning

SARSA

Behavior:
```
Experiment 1 (1a/1b/1c): GUI forces Q-Learning (SARSA option disabled).

Experiment 2: GUI forces SARSA (Q-Learning disabled).

Experiments 3 & 4: both options are available.

Q-Learning → replicates Exp 1.c behavior.

SARSA → replicates Exp 2 behavior.
```
d) Animation toogle
Turn on or off animation output which include Greedy vs BFS, Heatmap, Quiver map, and animation
*Note: If turn on, it may take a while to complete a run because animation rendering


e) Seeds (F / M)

Two entry fields:
```
Seed (F): – for Agent F

Seed (M): – for Agent M
```
You can enter:
```
A single integer (e.g. 111)

Or multiple integers separated by commas (e.g. 111, 222)

The GUI passes these to run_experiment.py as:

--seedF 111 222
--seedM 333 444
```

⚠️ Important:

The number of seeds for F and M must be at least equal to the number in Runs.

Example: if Runs = 2, you need at least 2 seeds for F and 2 seeds for M.

If not, the GUI will show an error and won’t start the experiment.

**4. Running an Experiment**

Configure the fields in Configuration:

Choose Experiment (1a–4)

Set Runs

Select Algorithm (if allowed by that experiment)

Enter seeds for F and M:

Example:
```
Seed (F): 111, 222

Seed (M): 333, 444
```
Click “Run Experiment”

What happens:

The Run button becomes disabled during execution (to prevent multiple overlapping runs).

The Live Output area is cleared and a message like Starting experiment... is shown.

The GUI launches a subprocess:

python -u run_experiment.py <experiment> --runs <R> --algo <algo> --seedF ... --seedM ...


Output from the subprocess is streamed back into the Live Output window line by line.

When the experiment finishes:

A line like --- Experiment Finished --- appears.

The Run button is re-enabled.

**5. Live Output**
The Live Output section is a scrollable text area that shows everything printed by:

run_experiment.py

the individual experiment scripts (exp1.py, exp2.py, exp3.py, exp4.py)

any log messages from visualization helpers

You will typically see:
```
Launch message from run_experiment.py, e.g.:

--- Launching Experiment 1c ---

Per-run info, e.g.:

Run 1/2 -> artifacts/exp1c_run1

Step/episode heartbeats you added, like:

STEP 500/8000 | ep=3 | agent=M | pol=PEXPLOIT | ...

Episode termination / pickup change messages, e.g.:

[TERM] episode 3 ended (terminals=3/6, steps=1324)

[EVENT] PICKUPS_CHANGED at ep=3 → [[1, 2, 1], [4, 5, 1]]
```
Summary lines like:
```
[OK] Wrote: artifacts/exp1c_run1

[exp3] Done. See artifacts/exp3-...

[exp4] Done. See artifacts/exp4-...
```
You can scroll back through this log while the experiment runs or after it has finished.

**6. Output Files & Where to Find Results**

All experiments write artifacts to the artifacts/ directory by default.
```
Exp 1 (1a, 1b, 1c)
artifacts/exp1a_run1
artifacts/exp1b_run2
etc.
```
Inside each run folder:
```
steps.csv – per-step log

episodes.csv – per-episode statistics

episodes_from_steps.csv – derived episode stats

learning_curve.png – reward curves

coordination.png – coordination metrics

qtable_F.json / qtable_M.json – Q-tables
```
Exp 2
```
artifacts/exp2_run1, artifacts/exp2_run2, …
```
Exp 3 (alpha sweep)
```
A timestamped batch folder like:

artifacts/exp3-YYYYMMDD-HHMMSS/

Per-alpha per-run subfolders

summary.csv, summary.md
```
Exp 4 (pickup change)
```
A timestamped batch folder like:

artifacts/exp4-YYYYMMDD-HHMMSS/

Per-run subfolders

summary.csv, summary.md

Visualizations (if enabled) are often placed in a viz/ subfolder.
```
Result Files:

Each experiment folder contains:

File	Description
```
steps.csv	Step-by-step environment log
episodes.csv	Episode-level summaries
episodes_from_steps.csv	Derived metrics
learning_curve.png	Reward curve
coordination.png	Coordination metrics
qtable_F.json / qtable_M.json	Final Q-tables
viz/Heatmaps, quiver plots, or animations if visualization is enabled
```

# Method 2: Running Directly from the Experiment Scripts (Backup)

If you prefer command-line execution or the GUI is unavailable, you can invoke each experiment script manually.
Each file supports --help for usage info.

**Experiment 1 — Q-Learning (1a, 1b, 1c)**
```
python experiment/exp1.py --variant a --runs 2 --alpha 0.3 --gamma 0.5 \
    --seedF 111 222 --seedM 333 444 --outroot artifacts
```

Variants:
```

--variant a → PRANDOM → PRANDOM

--variant b → PRANDOM → PGREEDY

--variant c → PRANDOM → PEXPLOIT
```
Each run creates:
```
artifacts/exp1a_run1/
artifacts/exp1a_run2/
```
**Experiment 2 — SARSA**
```
python experiment/exp2.py --runs 2 --alpha 0.3 --gamma 0.5 \
    --seedF 135 246 --seedM 864 975 --outroot artifacts
```

This automatically runs the on-policy SARSA agent (PEXPLOIT after warmup).

**Experiment 3 — Alpha Sweep (Q-Learning or SARSA)**
Q-Learning version (default)
```
python experiment/exp3.py --algo qlearning --alphas 0.15 0.45 --runs 2 \
    --seedF 101 202 --seedM 303 404 --outroot artifacts
```
SARSA version
```
python experiment/exp3.py --algo sarsa --alphas 0.15 0.45 --runs 2 \
    --seedF 111 222 --seedM 333 444 --outroot artifacts
```

Creates a timestamped folder such as:
```
artifacts/exp3-20251107-120000/
  ├── qlearning_a015_run1/
  ├── qlearning_a045_run2/
  ├── summary.csv
  └── summary.md
```
**Experiment 4 — Pickup Change Mid-Run**
Q-Learning
```
python experiment/exp4.py --algo qlearning --runs 2 \
    --seedF 111 211 --seedM 333 433 --outroot artifacts
```
SARSA
```
python experiment/exp4.py --algo sarsa --runs 2 \
    --seedF 135 246 --seedM 864 975 --outroot artifacts
```
