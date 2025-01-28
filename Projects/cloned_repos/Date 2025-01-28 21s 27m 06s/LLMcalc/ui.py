from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
                               QComboBox, QToolTip)
import LLMcalc

class LLMCalculatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Calculator")
        self.setMinimumSize(1000, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Model input section
        model_layout = QHBoxLayout()
        model_label = QLabel("Model ID:")
        self.model_input = QLineEdit("microsoft/phi-4")  # Default example
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_input)
        layout.addLayout(model_layout)

        # Hardware override section
        hw_layout = QHBoxLayout()

        vram, bandwidth, num_gpus, brand = LLMcalc.get_vram_specs()

        if not vram: vram = 0 # Just in case
        if not bandwidth: bandwidth = 0

        vram_label = QLabel("VRAM (GB):")
        self.vram_input = QLineEdit(str(vram))

        bandwidth_label = QLabel("Bandwidth (GB/s):")
        self.bandwidth_input = QLineEdit(str(bandwidth))

        gpu_count_label = QLabel("GPU Count:")
        self.gpu_count = QComboBox()
        self.gpu_count.addItems([str(i) for i in range(1, 16)])
        self.gpu_count.setCurrentIndex(num_gpus-1)

        hw_layout.addWidget(vram_label)
        hw_layout.addWidget(self.vram_input)
        hw_layout.addWidget(bandwidth_label)
        hw_layout.addWidget(self.bandwidth_input)
        hw_layout.addWidget(gpu_count_label)
        hw_layout.addWidget(self.gpu_count)
        layout.addLayout(hw_layout)

        # Calculate button
        self.calc_button = QPushButton("Calculate")
        self.calc_button.clicked.connect(self.calculate)
        layout.addWidget(self.calc_button)

        # Results table
        self.results_table = QTableWidget()
        layout.addWidget(self.results_table)

        # System info display
        self.system_info = QLabel()
        layout.addWidget(self.system_info)
        self.update_system_info()

    def update_system_info(self, real_bandwidth=None, real_num_gpus=None):
        """Update system information display"""
        total_ram = LLMcalc.get_ram_specs()
        vram, bandwidth, num_gpus, brand = LLMcalc.get_vram_specs()
        ram_bandwidth = LLMcalc.get_memory_bandwidth()

        if real_num_gpus: num_gpus = real_num_gpus
        if real_bandwidth: bandwidth = real_bandwidth
        if not vram: vram = 0
        if not bandwidth: bandwidth = 0

        info = f"System Info: RAM {total_ram:.2f} GB | RAM BW {ram_bandwidth:.2f} GB/s | VRAM {num_gpus}x{vram:.2f} GB | GPU BW ~{bandwidth:.1f} GB/s | GPU: {brand} "
        self.system_info.setText(info)

    def calculate(self):
        """Perform calculation and display results in table"""
        self.results_table.clearContents()

        model_id = self.model_input.text().strip()
        if not model_id:
            self.system_info.setText("Please enter a model ID")
            return
        self.config_data = LLMcalc.fetch_model_config(model_id)

        params_text = LLMcalc.get_model_params(model_id)
        if not params_text:
            self.system_info.setText("Could not determine model parameters")
            return

        params_b = LLMcalc.convert_params_to_b(params_text)
        total_ram = LLMcalc.get_ram_specs()
        vram, bandwidth, num_gpus, brand = LLMcalc.get_vram_specs()
        ram_bandwidth = LLMcalc.get_memory_bandwidth()

        # Apply overrides
        try:
            vram = float(self.vram_input.text()) if self.vram_input.text().strip() else vram
            bandwidth = float(self.bandwidth_input.text()) if self.bandwidth_input.text().strip() else bandwidth
            gpu_count = int(self.gpu_count.currentText())

            coef = 1
            adj_bandwidth = 0
            for i in range(gpu_count):
                adj_bandwidth += bandwidth * coef
                coef = 0.42
            bandwidth = adj_bandwidth
            vram *= gpu_count

        except ValueError:
            self.system_info.setText("Invalid number format in overrides")
            return

        self.update_system_info(bandwidth, gpu_count)

        results = LLMcalc.analyze_all_quantizations(params_b, vram, bandwidth, total_ram, ram_bandwidth, self.config_data)

        quant_levels = list(results.keys())
        run_order = ["All in VRAM", "KV cache offload", "Partial offload", "All in System RAM"]
        self.results_table.setRowCount(len(run_order) + 1)  # +1 for required memory
        self.results_table.setColumnCount(len(quant_levels))

        # Set headers
        self.results_table.setHorizontalHeaderLabels( quant_levels)
        self.results_table.setVerticalHeaderLabels(run_order + ["Required Memory (GB)"])

        # Fill table
        for col, quant in enumerate(quant_levels):
            for row, run_type in enumerate(run_order):
                if results[quant]['run_type'] == run_type:
                    tk_s = results[quant]['tk/s']
                    offload_pct = 100 - results[quant]['offload_percentage']
                    cell = QTableWidgetItem(f"{tk_s:.2f} tk/s")
                    cell.setToolTip(f"{offload_pct:.1f}% Layers Offloaded, {round(results[quant]['context'])}tk max context length")
                    self.results_table.setItem(row, col, cell)

            # Memory required row
            mem_cell = QTableWidgetItem(f"{results[quant]['memory_required']:.2f} GB")
            self.results_table.setItem(len(run_order), col, mem_cell)
