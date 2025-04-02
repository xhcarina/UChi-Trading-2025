import tkinter as tk
from tkinter import ttk
import json

class ParametersEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parameters Editor")

        # Create a style to tweak colors
        self.style = ttk.Style(self)
        self.style.theme_use("default")
        self.style.configure("TLabelFrame", background="#EAEAEA")   # frame background
        self.style.configure("TFrame", background="#EAEAEA")        # main background
        self.style.configure("TLabel", background="#EAEAEA")        # label background
        self.style.configure("TButton", background="#DDD")          # button background

        # Container for everything
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Create frames
        self.top_frame = ttk.LabelFrame(container, text="General Settings")
        self.etf_frame = ttk.LabelFrame(container, text="ETF Parameters")
        self.contract_frame = ttk.LabelFrame(container, text="Contract Parameters")
        self.bottom_frame = ttk.Frame(container)

        self.top_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.etf_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.contract_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.bottom_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        container.rowconfigure(2, weight=1)
        container.rowconfigure(3, weight=0)

        # Initialize a dictionary to store parameter values
        self.params = {
            "max_position": [200, 50, 100, 150, 200],
            "etf_min_margin": 1.0,
            "etf_fade": [10, 10, 20, 30],
            "etf_edge_sens": [1.5, 0.5, 1, 1.5],
            "etf_slack": [3, 1, 2, 3],
            "contract_min_margin": [80, 5, 10, 20, 40, 80],
            "contract_fade": [0.25, 0.1, 0.25, 0.5, 1],
            "contract_slack": [4, 2, 3, 4],
            "spreads": ["[2,4,6]", "[2,4,6]", "[5,10,15]", "[10,20,30]"],
            "level_orders": [3, 1, 2, 3],
            "etf_margin": [120, 60, 80, 100, 120],
            "safety": True
        }

        # Build the sections
        self.build_top_frame()
        self.build_etf_frame()
        self.build_contract_frame()
        self.build_bottom_frame()

    def build_top_frame(self):
        # Example: Max Position row
        tk.Label(self.top_frame, text="Max Position:").grid(row=0, column=0, sticky="e")
        self.max_position_entries = []
        for i, val in enumerate(self.params["max_position"]):
            e = tk.Entry(self.top_frame, width=6)
            e.insert(0, val)
            e.grid(row=0, column=i+1, padx=2)
            self.max_position_entries.append(e)

    def build_etf_frame(self):
        # Row for Min Margin
        row_idx = 0
        ttk.Label(self.etf_frame, text="Min Margin:").grid(row=row_idx, column=0, sticky="e")
        self.etf_min_margin_entry = tk.Entry(self.etf_frame, width=6)
        self.etf_min_margin_entry.insert(0, self.params["etf_min_margin"])
        self.etf_min_margin_entry.grid(row=row_idx, column=1, padx=2)

        # Row for Fade
        row_idx += 1
        ttk.Label(self.etf_frame, text="Fade:").grid(row=row_idx, column=0, sticky="e")
        self.etf_fade_entries = []
        for i, val in enumerate(self.params["etf_fade"]):
            e = tk.Entry(self.etf_frame, width=6)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.etf_fade_entries.append(e)

        # Row for Edge Sensitivity
        row_idx += 1
        ttk.Label(self.etf_frame, text="Edge Sensitivity:").grid(row=row_idx, column=0, sticky="e")
        self.etf_edge_entries = []
        for i, val in enumerate(self.params["etf_edge_sens"]):
            e = tk.Entry(self.etf_frame, width=6)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.etf_edge_entries.append(e)

        # Row for Slack
        row_idx += 1
        ttk.Label(self.etf_frame, text="Slack:").grid(row=row_idx, column=0, sticky="e")
        self.etf_slack_entries = []
        for i, val in enumerate(self.params["etf_slack"]):
            e = tk.Entry(self.etf_frame, width=6)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.etf_slack_entries.append(e)

    def build_contract_frame(self):
        # Row for Min Margin
        row_idx = 0
        ttk.Label(self.contract_frame, text="Min Margin:").grid(row=row_idx, column=0, sticky="e")
        self.contract_min_margin_entries = []
        for i, val in enumerate(self.params["contract_min_margin"]):
            e = tk.Entry(self.contract_frame, width=6)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.contract_min_margin_entries.append(e)

        # Row for Fade
        row_idx += 1
        ttk.Label(self.contract_frame, text="Edge Sensitivity:").grid(row=row_idx, column=0, sticky="e")
        self.contract_fade_entries = []
        for i, val in enumerate(self.params["contract_fade"]):
            e = tk.Entry(self.contract_frame, width=6)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.contract_fade_entries.append(e)

        # Row for Slack
        row_idx += 1
        ttk.Label(self.contract_frame, text="Slack:").grid(row=row_idx, column=0, sticky="e")
        self.contract_slack_entries = []
        for i, val in enumerate(self.params["contract_slack"]):
            e = tk.Entry(self.contract_frame, width=6)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.contract_slack_entries.append(e)

        # Row for Spreads
        row_idx += 1
        ttk.Label(self.contract_frame, text="Spreads:").grid(row=row_idx, column=0, sticky="e")
        self.spreads_entries = []
        for i, val in enumerate(self.params["spreads"]):
            e = tk.Entry(self.contract_frame, width=8)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.spreads_entries.append(e)

        # Row for Level Orders
        row_idx += 1
        ttk.Label(self.contract_frame, text="Level Orders:").grid(row=row_idx, column=0, sticky="e")
        self.level_orders_entries = []
        for i, val in enumerate(self.params["level_orders"]):
            e = tk.Entry(self.contract_frame, width=6)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.level_orders_entries.append(e)

        # Row for ETF Margin
        row_idx += 1
        ttk.Label(self.contract_frame, text="ETF Margin:").grid(row=row_idx, column=0, sticky="e")
        self.etf_margin_entries = []
        for i, val in enumerate(self.params["etf_margin"]):
            e = tk.Entry(self.contract_frame, width=6)
            e.insert(0, val)
            e.grid(row=row_idx, column=i+1, padx=2)
            self.etf_margin_entries.append(e)

    def build_bottom_frame(self):
        # Safety Buttons
        ttk.Label(self.bottom_frame, text="Safety:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.safety_var = tk.BooleanVar(value=self.params["safety"])

        # True button
        true_btn = ttk.Radiobutton(
            self.bottom_frame,
            text="True",
            variable=self.safety_var,
            value=True
        )
        true_btn.grid(row=0, column=1, padx=2)

        # False button
        false_btn = ttk.Radiobutton(
            self.bottom_frame,
            text="False",
            variable=self.safety_var,
            value=False
        )
        false_btn.grid(row=0, column=2, padx=2)

        # Update button
        update_btn = ttk.Button(self.bottom_frame, text="Update", command=self.on_update)
        update_btn.grid(row=0, column=3, padx=10, pady=5)

        # Save button (optional) to store in a JSON config
        save_btn = ttk.Button(self.bottom_frame, text="Save", command=self.on_save)
        save_btn.grid(row=0, column=4, padx=10, pady=5)

    def on_update(self):
        # Update max_position
        new_max_pos = []
        for e in self.max_position_entries:
            try:
                new_max_pos.append(float(e.get()))
            except ValueError:
                new_max_pos.append(0)
        self.params["max_position"] = new_max_pos

        # Update ETF Min Margin
        try:
            self.params["etf_min_margin"] = float(self.etf_min_margin_entry.get())
        except ValueError:
            self.params["etf_min_margin"] = 0.0

        # Update ETF Fade
        new_etf_fade = []
        for e in self.etf_fade_entries:
            try:
                new_etf_fade.append(float(e.get()))
            except ValueError:
                new_etf_fade.append(0)
        self.params["etf_fade"] = new_etf_fade

        # Update ETF Edge Sensitivity
        new_etf_edge = []
        for e in self.etf_edge_entries:
            try:
                new_etf_edge.append(float(e.get()))
            except ValueError:
                new_etf_edge.append(0)
        self.params["etf_edge_sens"] = new_etf_edge

        # Update ETF Slack
        new_etf_slack = []
        for e in self.etf_slack_entries:
            try:
                new_etf_slack.append(float(e.get()))
            except ValueError:
                new_etf_slack.append(0)
        self.params["etf_slack"] = new_etf_slack

        # Update Contract Min Margin
        new_contract_min_margin = []
        for e in self.contract_min_margin_entries:
            try:
                new_contract_min_margin.append(float(e.get()))
            except ValueError:
                new_contract_min_margin.append(0)
        self.params["contract_min_margin"] = new_contract_min_margin

        # Update Contract Fade (Edge Sensitivity in contract section)
        new_contract_fade = []
        for e in self.contract_fade_entries:
            try:
                new_contract_fade.append(float(e.get()))
            except ValueError:
                new_contract_fade.append(0)
        self.params["contract_fade"] = new_contract_fade

        # Update Contract Slack
        new_contract_slack = []
        for e in self.contract_slack_entries:
            try:
                new_contract_slack.append(float(e.get()))
            except ValueError:
                new_contract_slack.append(0)
        self.params["contract_slack"] = new_contract_slack

        # Update Spreads (kept as strings in this example)
        new_spreads = []
        for e in self.spreads_entries:
            new_spreads.append(e.get())
        self.params["spreads"] = new_spreads

        # Update Level Orders
        new_level_orders = []
        for e in self.level_orders_entries:
            try:
                new_level_orders.append(float(e.get()))
            except ValueError:
                new_level_orders.append(0)
        self.params["level_orders"] = new_level_orders

        # Update ETF Margin
        new_etf_margin = []
        for e in self.etf_margin_entries:
            try:
                new_etf_margin.append(float(e.get()))
            except ValueError:
                new_etf_margin.append(0)
        self.params["etf_margin"] = new_etf_margin

        # Update Safety (True/False)
        self.params["safety"] = self.safety_var.get()

        # Print all updated parameters for debugging
        print("Updated parameters:", self.params)

    def on_save(self):
        """
        Example: Save current params to a JSON file so you can
        load them next time you run the GUI or your bot.
        """
        self.on_update()  # first gather all current values
        with open("params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        print("Parameters saved to params.json")

if __name__ == "__main__":
    app = ParametersEditor()
    app.mainloop()
