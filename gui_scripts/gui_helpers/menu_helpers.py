# pylint: disable=c-extension-no-member

from PyQt5 import QtWidgets


class MenuHelpers:
    """
    Contains methods related to setting up the menu and their potential options.
    """

    def __init__(self):
        self.menu_bar_obj = None  # Updated from run_gui.py script

        self.file_menu_obj = None
        self.help_menu_obj = None
        self.edit_menu_obj = None

    def open_file(self):
        """
        Opens a json, yaml, or ini file.
        """
        # Set the file dialog to filter for .yml and .json files only
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.menu_bar_obj, "Open Configuration File", "", "Config Files (*.yml *.json *.ini)"
        )
        if file_name:
            print(f"Selected file: {file_name}")

    def create_file_menu(self):
        """
        Creates the basis of the file menu along with adding an open action.
        """
        self.file_menu_obj = self.menu_bar_obj.addMenu('&File')
        open_action = QtWidgets.QAction('&Load Configuration from File', self.menu_bar_obj)
        open_action.triggered.connect(self.open_file)
        self.file_menu_obj.addAction(open_action)

    def create_edit_menu(self):
        """
        Creates the edit menu section.
        """
        self.edit_menu_obj = self.menu_bar_obj.addMenu('&Edit')

    def create_help_menu(self):
        """
        Creates the help menu section.
        """
        self.help_menu_obj = self.menu_bar_obj.addMenu('&Help')
