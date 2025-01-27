from config import CONFIG
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon


class SystemTray(QSystemTrayIcon):
    def __init__(self):
        icon = CONFIG.get("app").get("icon")

        super().__init__(QIcon(icon))

        # Create the tray menu
        self.menu = QMenu()

        # Add actions to the menu
        self.create_menu_actions()

        # Set the menu to the tray icon
        self.setContextMenu(self.menu)

    def create_menu_actions(self):
        """Creates and adds actions to the context menu."""

        # Action to quit the application
        quit_action = QAction("❌ Quit", self)
        quit_action.triggered.connect(self.quit_app)
        quit_action.setToolTip("Quit the application")

        # Add actions to the menu
        self.menu.addAction(quit_action)

    def open_settings(self):
        """Placeholder for a settings dialog."""
        self.show_message("Settings", "Settings window would appear here.")

    def show_message(self, title: str, message: str):
        """Displays a message balloon from the system tray."""
        self.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information)

    def quit_app(self):
        """Exits the application."""
        QCoreApplication.quit()
