# pylint: disable=no-name-in-module
# pylint: disable=c-extension-no-member

import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileSystemModel, QTreeView, QTabWidget, QPlainTextEdit  # pylint: disable=unused-import

# Assuming these imports are available from your project
from gui_scripts.gui_helpers.menu_helpers import MenuHelpers
from gui_scripts.gui_helpers.action_helpers import ActionHelpers
from gui_scripts.gui_helpers.button_helpers import ButtonHelpers
from gui_scripts.gui_helpers.highlight_helpers import PythonHighlighter
from gui_scripts.gui_helpers.general_helpers import DirectoryTreeView
from gui_scripts.gui_args.style_args import STYLE_SHEET


class MainWindow(QtWidgets.QMainWindow):
    """
    The main window class, central point that controls all GUI functionality and actions.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SD-EON Simulator")
        self.resize(1280, 720)

        self.menu_help_obj = MenuHelpers()
        self.ac_help_obj = ActionHelpers()
        self.button_help_obj = ButtonHelpers()

        self.progress_bar = QtWidgets.QProgressBar()
        self.simulation_thread = None
        self.network_option = ''

        self.first_info_pane = None
        self.media_dir = None
        self.settings_button = None
        self.mw_toolbar = None

        self.current_file_path = None
        self.main_widget = None
        self.main_layout = None
        self.horizontal_splitter = None
        self.first_info_layout = None
        self.directory_tree_obj = None
        self.tab_widget = None
        self.file_editor = None
        self.mw_topology_view_area = None
        self.vertical_splitter = None
        self.bottom_pane = None
        self.menu_bar = None
        self.tool_bar = None
        self.status_bar = None
        self.highlighter = None

        # Set the project directory as the root for the file model
        self.project_directory = QtCore.QDir.currentPath()
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.project_directory)

        self.init_ui()
        self.init_menu_bar()
        self.init_tool_bar()
        self.init_status_bar()
        self.apply_styles()

    def init_ui(self):
        """
        Initialize the main user-interface.
        """
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Remove any margins
        self.main_layout.setSpacing(0)  # Remove any spacing between widgets

        # Splitter for directory tree and tab widget
        self.horizontal_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Directory tree view
        self.first_info_pane = QtWidgets.QWidget()
        self.first_info_layout = QtWidgets.QVBoxLayout(self.first_info_pane)

        self.directory_tree_obj = DirectoryTreeView(self.file_model)
        self.directory_tree_obj.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.directory_tree_obj.customContextMenuRequested.connect(
            self.directory_tree_obj.handle_context_menu)
        # self.directory_tree.setModel(self.file_model)
        self.directory_tree_obj.setRootIndex(self.file_model.index(self.project_directory))

        # connect directory tree view signals to slots
        self.directory_tree_obj.item_double_clicked_sig.connect(self.on_tree_item_dclicked)

        # Apply custom stylesheet for font size
        self.directory_tree_obj.setStyleSheet("font-size: 12pt;")  # Increase font size to 12pt

        # Hide headers, size, type, and date columns
        self.directory_tree_obj.setHeaderHidden(True)
        self.directory_tree_obj.setColumnHidden(1, True)  # Size column
        self.directory_tree_obj.setColumnHidden(2, True)  # Type column
        self.directory_tree_obj.setColumnHidden(3, True)  # Date column

        self.first_info_layout.addWidget(self.directory_tree_obj)
        self.horizontal_splitter.addWidget(self.first_info_pane)

        # Tab widget for file editor and topology
        self.tab_widget = QTabWidget()

        # File editor tab
        self.file_editor = QtWidgets.QTextEdit()
        self.file_editor.setObjectName("src_code_editor")
        self.file_editor.setStyleSheet("font-size: 12pt;")
        self.tab_widget.addTab(self.file_editor, "File Editor")

        # Apply Python syntax highlighter to the file editor
        self.highlighter = PythonHighlighter(self.file_editor.document())

        # Topology view tab
        self.mw_topology_view_area = QtWidgets.QScrollArea()
        init_topology_data = QtWidgets.QLabel("Nothing to display", self.mw_topology_view_area)
        init_topology_data.setStyleSheet("font-size: 11pt")
        init_topology_data.setAlignment(QtCore.Qt.AlignCenter)
        self.mw_topology_view_area.setWidget(init_topology_data)
        self.mw_topology_view_area.setAlignment(QtCore.Qt.AlignCenter)
        self.tab_widget.setStyleSheet("font-size: 12pt")
        self.tab_widget.addTab(self.mw_topology_view_area, "Topology View")

        self.horizontal_splitter.addWidget(self.tab_widget)

        # Vertical splitter for the horizontal splitter and the bottom pane
        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vertical_splitter.addWidget(self.horizontal_splitter)

        # Bottom pane for simulation output
        self.bottom_pane = QPlainTextEdit(self)

        self.bottom_pane.setCursor(QtCore.Qt.ArrowCursor)  # Uses the pointer cursor
        self.bottom_pane.viewport().setCursor(QtCore.Qt.ArrowCursor)

        self.bottom_pane.setReadOnly(True)
        self.bottom_pane.setMinimumHeight(150)
        self.vertical_splitter.addWidget(self.bottom_pane)

        self.main_layout.addWidget(self.vertical_splitter, stretch=1)

    def on_tree_item_dclicked(self, index):
        """
        Performs an action when treeview is double-clicked.

        :param index: Index of file path displayed in the tree.
        """
        file_path = self.file_model.filePath(index)
        if QtCore.QFileInfo(file_path).isFile():
            file_index = self.tab_widget.indexOf(self.file_editor)
            self.tab_widget.setTabText(file_index, QtCore.QFileInfo(file_path).fileName())
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.file_editor.setPlainText(content)
            self.current_file_path = file_path
            self.tab_widget.setCurrentWidget(self.file_editor)

    def save_file(self):
        """
        Saves a file edited.
        """
        if hasattr(self, 'current_file_path') and self.current_file_path:
            with open(self.current_file_path, 'w', encoding='utf-8') as file:
                file.write(self.file_editor.toPlainText())

    def init_menu_bar(self):
        """
        Initialize the menu bar.
        """
        self.menu_bar = self.menuBar()
        self.menu_help_obj.menu_bar_obj = self.menu_bar
        self.menu_help_obj.create_file_menu()
        self.menu_help_obj.create_edit_menu()
        self.menu_help_obj.create_help_menu()

        self.ac_help_obj.mw_topology_view_area = self.mw_topology_view_area
        self.ac_help_obj.menu_help_obj = self.menu_help_obj
        self.ac_help_obj.menu_bar_obj = self.menu_bar

        self.ac_help_obj.create_topology_action()
        self.ac_help_obj.create_save_action()

        # Create exit action and add it to the File menu
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        self.menu_help_obj.file_menu_obj.addAction(exit_action)

        self.ac_help_obj.create_settings_action()
        self.ac_help_obj.create_about_action()

    def init_tool_bar(self):
        """
        Initialize the toolbar.
        """
        self.tool_bar = self.addToolBar('Main Toolbar')
        self.tool_bar.setMovable(False)
        self.tool_bar.setIconSize(QtCore.QSize(15, 15))

        save_action = QtWidgets.QAction('Save', self)
        save_action.triggered.connect(self.save_file)
        self.tool_bar.addAction(save_action)

        self.button_help_obj.bottom_right_pane = self.bottom_pane  # Ensure this points to QPlainTextEdit
        self.button_help_obj.progress_bar = self.progress_bar
        self.button_help_obj.create_settings_button()
        self.button_help_obj.create_start_button()
        self.button_help_obj.create_stop_button()
        self.button_help_obj.create_pause_button()

        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.button_help_obj.start_button)
        self.tool_bar.addAction(self.button_help_obj.pause_button)
        self.tool_bar.addAction(self.button_help_obj.stop_button)
        self.tool_bar.addSeparator()
        self.tool_bar.addWidget(self.button_help_obj.settings_button)

    def init_status_bar(self):
        """
        Initialize the status bar.
        """
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Active')
        self.status_bar.addWidget(self.progress_bar)
        self.progress_bar.setVisible(False)

    def apply_styles(self):
        """
        Apply styles to the display.
        """
        self.setStyleSheet(STYLE_SHEET)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
