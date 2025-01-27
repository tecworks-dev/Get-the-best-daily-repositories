import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QHideEvent, QShowEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

LOG = logging.getLogger(__name__)


class TextWindow(QWidget):
    custom_signal = pyqtSignal(str, str)
    windows_event = pyqtSignal(bool)
    process_text_event = pyqtSignal(str, str, str)
    event_put_app_focus = pyqtSignal()

    def __init__(self, processing_text, functions_list):
        super().__init__()
        LOG.debug(f"Creating text window processing text, {processing_text}")
        # Layout for the main window
        main_layout = QHBoxLayout(self)

        self.processing_text = processing_text

        # Create QTabWidget for text areas
        self.text_tab_widget = QTabWidget()
        self.functions_list = functions_list
        # Add initial text tab
        self.text_tab = QWidget()
        text_layout = QVBoxLayout(self.text_tab)
        self.text_widget = QTextEdit()
        self.text_widget.setText(self.processing_text)
        text_layout.addWidget(self.text_widget)
        self.text_tab_widget.addTab(self.text_tab, "Text 1")

        # Create QTabWidget for functions list
        self.function_tab_widget = QTabWidget()

        # Tab 1 - Function list
        self.function_list_tab = QWidget()
        function_layout = QVBoxLayout(self.function_list_tab)
        self.function_list_widget = QListWidget()
        function_layout.addWidget(self.function_list_widget)
        self.function_tab_widget.addTab(self.function_list_tab, "Functions 1")

        # Add function names to the list
        for function_name in functions_list:
            self.function_list_widget.addItem(function_name)

        # Set the function list widget to be selectable
        self.function_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Tab 2 - Text area for functions
        self.function_text_tab = QWidget()
        function_text_layout = QVBoxLayout(self.function_text_tab)
        function_text_widget = QTextEdit()
        function_text_layout.addWidget(function_text_widget)
        self.function_tab_widget.addTab(self.function_text_tab, "Function Text")

        # Split the main layout into two sections: one for text and one for functions
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.text_tab_widget)
        splitter.addWidget(self.function_tab_widget)

        # Set stretch factors to ensure text and function sections take 90% of the space
        splitter.setStretchFactor(0, 9)  # Text section takes 9 parts
        splitter.setStretchFactor(1, 9)  # Function section takes 9 parts

        # Add splitter to the main layout
        main_layout.addWidget(splitter)

        # Bottom input area with 3 text inputs and buttons, occupying 10% of the horizontal space
        bottom_layout = QVBoxLayout()

        # Create a container frame for the bottom section to control its width
        bottom_frame = QFrame()
        bottom_frame.setLayout(bottom_layout)

        # Create 3 text input fields and 3 run buttons, and add them vertically
        self.inputs = [QLineEdit() for _ in range(3)]
        self.run_buttons = [QPushButton("Run") for _ in range(3)]

        for input_field, button in zip(self.inputs, self.run_buttons):
            bottom_layout.addWidget(input_field)
            bottom_layout.addWidget(button)

        # Add the bottom frame to the main layout, occupying 10% of the horizontal space
        bottom_frame.setFixedWidth(self.width() // 10)  # Set the width to 10% of the window size
        main_layout.addWidget(bottom_frame)

        self.setLayout(main_layout)
        self.setWindowTitle("Text Display and Function List")
        self.resize(800, 600)

        # Connect context menus for adding new tabs
        self.text_tab_widget.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.text_tab_widget.tabBar().customContextMenuRequested.connect(self.showTextTabMenu)

        self.function_tab_widget.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.function_tab_widget.tabBar().customContextMenuRequested.connect(self.showFunctionTabMenu)

        # Connect the Enter key event to the function application
        self.function_list_widget.keyPressEvent = self.handle_function_key_event

    def handle_function_key_event(self, event):
        if event.key() == Qt.Key.Key_Return:  # Detect Enter key press
            # Get the selected function name from the list
            selected_item = self.function_list_widget.currentItem()
            if selected_item:
                function_name = selected_item.text()
                # Apply the function to the displayed text
                self.apply_function(function_name)
        else:
            super().keyPressEvent(event)  # Pass the event to the base class if it's not Enter

    def apply_function(self, function_name):
        # Check if the function name exists in the functions list

        if function_name in self.functions_list:
            # Apply the function to the current text in the text widget
            func = self.functions_list[function_name]
            new_text = func(self.text_widget.toPlainText())
            self.text_widget.setText(new_text)  # Update the text widget with the result

    def showTextTabMenu(self, pos):
        # Create and show context menu for adding new text tabs
        menu = self.text_tab_widget.tabBar().createStandardContextMenu()
        add_tab_action = menu.addAction("Add New Text Tab")
        add_tab_action.triggered.connect(self.add_new_text_tab)
        menu.exec(self.text_tab_widget.tabBar().mapToGlobal(pos))

    def showFunctionTabMenu(self, pos):
        # Create and show context menu for adding new function tabs
        menu = self.function_tab_widget.tabBar().createStandardContextMenu()
        add_tab_action = menu.addAction("Add New Function Tab")
        add_tab_action.triggered.connect(self.add_new_function_tab)
        menu.exec(self.function_tab_widget.tabBar().mapToGlobal(pos))

    def add_new_text_tab(self):
        # Create new text tab
        new_text_tab = QWidget()
        new_text_layout = QVBoxLayout(new_text_tab)
        new_text_widget = QTextEdit()
        new_text_layout.addWidget(new_text_widget)

        # Add the new text tab
        new_tab_index = self.text_tab_widget.addTab(new_text_tab, f"Text {self.text_tab_widget.count() + 1}")

        # Optionally set the new tab as the current tab
        self.text_tab_widget.setCurrentIndex(new_tab_index)

    def add_new_function_tab(self):
        # Create new function list tab
        new_function_list_tab = QWidget()
        new_function_list_layout = QVBoxLayout(new_function_list_tab)
        new_function_list_widget = QListWidget()
        new_function_list_widget.addItem("Uppercase")  # Default function item
        new_function_list_widget.addItem("Lowercase")
        new_function_list_layout.addWidget(new_function_list_widget)

        # Add the new function list tab
        new_function_list_tab_index = self.function_tab_widget.addTab(
            new_function_list_tab, f"Functions {self.function_tab_widget.count() + 1}"
        )

        # Create new function text tab
        new_function_text_tab = QWidget()
        new_function_text_layout = QVBoxLayout(new_function_text_tab)
        new_function_text_widget = QTextEdit()
        new_function_text_layout.addWidget(new_function_text_widget)

        # Add the new function text tab
        new_function_text_tab_index = self.function_tab_widget.addTab(
            new_function_text_tab, f"Function Text {self.function_tab_widget.count()}"
        )

        # Optionally set the new function list tab as the current tab
        self.function_tab_widget.setCurrentIndex(new_function_list_tab_index)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Hide the window instead of closing it to prevent application shutdown
        event.ignore()
        self.hide()
        # Update the key listener state when the window is hidden
        self.windows_event.emit(False)

    def hideEvent(self, event: QHideEvent) -> None:
        # Ensure the key listener flag is updated when the window is hidden
        self.windows_event.emit(False)

    def showEvent(self, event: QShowEvent) -> None:
        LOG.debug("Text window shown")
        super().showEvent(event)
        # Ensure the window is focused when shown
        self.raise_()  # Raise the window to the top
        self.activateWindow()  # Make the window active (focused)

        self.windows_event.emit(True)

        self.event_put_app_focus.emit()
