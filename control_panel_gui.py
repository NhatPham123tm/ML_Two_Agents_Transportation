#!/usr/bin/env python3
"""
A simple Tkinter-based GUI Control Panel for running RL experiments.

This script provides a graphical interface to the `run_experiment.py`
command-line tool. It allows users to select an experiment, configure
basic parameters, and view the live output of the experiment run.

Usage:
  python control_panel_gui.py
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import queue
import os
import sys

class ControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Experiment Control Panel")
        self.geometry("750x600")
        
        self.process = None
        self.output_queue = queue.Queue()
        self.tk.call("source", "Azure-ttk-theme-2.1.0/azure.tcl") # Credit: https://github.com/rdbende/Azure-ttk-theme/
        self.tk.call("set_theme", "light")
        self._create_widgets()
        self._layout_widgets()
        self.update_ui_state()

        # Start polling the output queue
        self.after(100, self.poll_output_queue)

    def change_theme(self):
        # NOTE: The theme's real name is azure-<mode>
        if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
            # Set light theme
            self.tk.call("set_theme", "light")
        else:
            # Set dark theme
            self.tk.call("set_theme", "dark")
    def _create_widgets(self):
        """Create all the UI widgets."""
        # --- Variables ---
        self.exp_var = tk.StringVar(value="1c")
        self.runs_var = tk.StringVar(value="2")
        self.algo_var = tk.StringVar(value="qlearning")
        self.seedF_var = tk.StringVar(value="123")
        self.seedM_var = tk.StringVar(value="321")
        # --- Frames ---
        self.config_frame = ttk.LabelFrame(self, text="Configuration")
        self.action_frame = ttk.Frame(self)
        self.output_frame = ttk.LabelFrame(self, text="Live Output")

        # --- Experiment Selection ---
        exp_label = ttk.Label(self.config_frame, text="Experiment:")
        self.exp_radios = []
        for val, text in [("1a", "1a"), ("1b", "1b"), ("1c", "1c"),
                           ("2", "2"), ("3", "3"), ("4", "4")]:
            radio = ttk.Radiobutton(
                self.config_frame, text=text, variable=self.exp_var,
                value=val, command=self.update_ui_state
            )
            self.exp_radios.append(radio)

        # --- Parameters ---
        runs_label = ttk.Label(self.config_frame, text="Number of Runs:")
        self.runs_entry = ttk.Entry(self.config_frame, textvariable=self.runs_var, width=5)

        algo_label = ttk.Label(self.config_frame, text="Algorithm:")
        self.q_radio = ttk.Radiobutton(
            self.config_frame, text="Q-Learning", variable=self.algo_var, value="qlearning"
        )
        self.sarsa_radio = ttk.Radiobutton(
            self.config_frame, text="SARSA", variable=self.algo_var, value="sarsa"
        )

        # --- Action Buttons ---
        self.run_button = ttk.Button(self.action_frame, text="Run Experiment",style='Accent.TButton', command=self.run_experiment)
        self.cancel_button = ttk.Button(self.action_frame, text="Cancel", command=self.cancel_run)
        self.cancel_button.pack(pady=5)
        self.cancel_button.config(state=tk.DISABLED)

        # --- Seed Entry ---
        seedF_label = ttk.Label(self.config_frame, text="Seed (F):")
        self.seedF_entry = ttk.Entry(self.config_frame, textvariable=self.seedF_var, width=8)

        seedM_label = ttk.Label(self.config_frame, text="Seed (M):")
        self.seedM_entry = ttk.Entry(self.config_frame, textvariable=self.seedM_var, width=8)
        # --- Output Display ---
        self.output_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, state=tk.DISABLED)
        

    def _layout_widgets(self):
        """Place widgets in the grid layout."""
        self.columnconfigure(0, weight=1) # Make the main column expandable
        self.rowconfigure(2, weight=1)

        # Configuration Frame
        self.config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.config_frame.columnconfigure(1, weight=0)
        
        # Experiment Radios
        ttk.Label(self.config_frame, text="Experiment:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        for i, radio in enumerate(self.exp_radios):
            radio.grid(row=0, column=i + 1, padx=2, sticky="w")

        # Parameters
        ttk.Label(self.config_frame, text="Runs:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.runs_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.config_frame, text="Algorithm:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.q_radio.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.sarsa_radio.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        # Seeds
        ttk.Label(self.config_frame, text="Seeds:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.seedF_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.seedM_entry.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        ttk.Label(self.config_frame, text="(F / M)").grid(row=3, column=3, padx=5, pady=5, sticky="w")

        # Action Frame
        self.action_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.run_button.pack(pady=5)

        # Output Frame
        self.output_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.output_frame.columnconfigure(0, weight=1)
        self.output_frame.rowconfigure(0, weight=1)
        self.output_text.grid(row=0, column=0, sticky="nsew")


    def update_ui_state(self):
        """Enable/disable widgets based on the selected experiment."""
        exp = self.exp_var.get()
        if exp.startswith("1"):
            self.algo_var.set("qlearning")
            self.q_radio.config(state=tk.DISABLED)
            self.sarsa_radio.config(state=tk.DISABLED)
        elif exp == "2":
            self.algo_var.set("sarsa")
            self.q_radio.config(state=tk.DISABLED)
            self.sarsa_radio.config(state=tk.DISABLED)
        else: # Exp 3 and 4
            self.q_radio.config(state=tk.NORMAL)
            self.sarsa_radio.config(state=tk.NORMAL)

    def cancel_run(self):
        if self.process and self.process.poll() is None:
            try:
                # cross-platform friendly terminate
                self.process.terminate()
            except Exception:
                pass
        self.cancel_button.config(state=tk.DISABLED)

    def run_experiment(self):
        """Construct and run the experiment command in a separate thread."""
        self.run_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Starting experiment...\n\n")
        self.output_text.config(state=tk.DISABLED)
        
        # Determine correct python executable based on OS for cross-platform compatibility
        python_executable = "python" if sys.platform == "win32" else "python3"
        seedF = self.seedF_var.get()
        seedM = self.seedM_var.get()
        try:
            int(seedF)
            int(seedM)
        except ValueError:
            messagebox.showerror("Invalid Seed", "Seeds must be integers.")
            return
        # Construct the command
        cmd = [
            python_executable, "-u", "run_experiment.py", self.exp_var.get(),
            "--runs", self.runs_var.get(),
            "--algo", self.algo_var.get(),
            "--seedF", seedF,
            "--seedM", seedM,
        ]

        # Run in a thread to keep the GUI responsive
        thread = threading.Thread(target=self._execute_command, args=(cmd,), daemon=True)
        thread.start()

    def _execute_command(self, cmd):
        """Worker function to run a subprocess and stream its output."""
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', bufsize=1
        )
        # Stream output line by line
        for line in iter(self.process.stdout.readline, ''):
            self.output_queue.put(line)
        self.process.stdout.close()
        self.process.wait()
        self.output_queue.put("\n--- Experiment Finished ---")
    
    def poll_output_queue(self):
        """Check for new output and update UI."""
        while not self.output_queue.empty():
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break

            # Append to output
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, line)
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)

            # Handle finish flag
            if "--- Experiment Finished ---" in line:
                self.run_button.config(state=tk.NORMAL)
                # snap to 100% if determinate, otherwise stop animation

        self.after(100, self.poll_output_queue)

if __name__ == "__main__":
    app = ControlPanel()
    app.mainloop()