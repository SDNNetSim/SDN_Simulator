# pylint: disable=c-extension-no-member

import networkx as nx
from PyQt5 import QtWidgets as qtw, QtCore as qtc

from data_scripts.structure_data import create_network

from gui_scripts.gui_helpers.topology_helpers import TopologyCanvas
from gui_scripts.gui_helpers.general_helpers import SettingsDialog
from gui_scripts.gui_args.config_args import GUI_DEFAULT_SETTINGS
from gui_scripts.gui_helpers.dialogs import Alert
from gui_scripts.gui_args.config_args import AlertCode



class MenuBar(qtw.QMenuBar):
    """
    The Menu Bar.
    """
    app_close_signal_relay = qtc.pyqtSignal()

    def __init__(self, main_window_ref: qtw.QMainWindow, parent=None):
        """
        Initializes the menu_bar.

        :param main_window_ref: reference to the main window on which it belongs
        :type QtWidgets.QMainWindow:
        """
        super().__init__(parent)
        self.main_window_ref_obj = main_window_ref
        self.menu_creator_obj = MenuCreator(self, self.main_window_ref_obj)

        self.file_menu_obj = None
        self.help_menu_obj = None
        self.edit_menu_obj = None
        self.plot_menu_obj = None

        self.setup_menu_bar()

    def setup_menu_bar(self):
        """
        Sets up the menu options for the menu bar.

        :param : None
        :return : None
        """
        self.file_menu_obj = self.menu_creator_obj.create_file_menu()
        self.help_menu_obj = self.menu_creator_obj.create_edit_menu()
        self.edit_menu_obj = self.menu_creator_obj.create_plot_menu()
        self.plot_menu_obj = self.menu_creator_obj.create_help_menu()


class MenuCreator:
    """
    Contains methods for configuring the menubar with menu items.
    """

    def __init__(self, menu_bar: MenuBar, main_window_ref: qtw.QMainWindow):
        """
        Initializes the menu helpers object.

        :param menu_bar: A reference to the main window menu bar
        :type menu_bar: QtWidgets.QMenuBar
        :return : None
        """
        self.menu_bar_obj = menu_bar
        self.main_window_ref_obj = main_window_ref
        self.menu_bar_action_handler_obj = MenuActionHandler(self.menu_bar_obj, self.main_window_ref_obj)

    def create_file_menu(self) -> qtw.QMenu:
        """
        Creates the basis of the file menu along with adding an open action.
        """
        # add file menu
        file_menu_obj = self.menu_bar_obj.addMenu('File')

        # create file menu actions
        load_config_action = self.menu_bar_action_handler_obj.create_load_config_from_file_action()
        display_topology_action = self.menu_bar_action_handler_obj.create_display_topology_action()
        settings_action = self.menu_bar_action_handler_obj.create_settings_action()
        exit_action = self.menu_bar_action_handler_obj.create_exit_action()

        # arrange file menu actions
        file_menu_obj.addAction(load_config_action)
        file_menu_obj.addAction(display_topology_action)
        file_menu_obj.addAction(settings_action)
        file_menu_obj.addSeparator()
        file_menu_obj.addAction(exit_action)

        return file_menu_obj

    def create_edit_menu(self) -> qtw.QMenu:
        """
        Creates the edit menu section.
        """
        edit_menu_obj = self.menu_bar_obj.addMenu('Edit')
        return edit_menu_obj

    def create_help_menu(self) -> qtw.QMenu:
        """
        Creates the help menu section.
        """
        help_menu_obj = self.menu_bar_obj.addMenu('Help')
        # add about action
        about_action = self.menu_bar_action_handler_obj.create_about_action()

        help_menu_obj.addAction(about_action)

        return help_menu_obj

    def create_plot_menu(self) -> qtw.QMenu:
        """
        Creates the plot menu
        """
        # add plot menu
        plot_menu_obj = self.menu_bar_obj.addMenu('Plot')

        # create plot menu actions e.g. plot >, configure plotting settings, export plot, etc
        plot_submenu_obj = qtw.QMenu('Plot', plot_menu_obj)

        plots = GUI_DEFAULT_SETTINGS['plots']
        for plot_action in plots:
            action = self.menu_bar_action_handler_obj.create_plot_sm_action(plot_action)
            action.setParent(plot_submenu_obj)
            plot_submenu_obj.addAction(action)

        config_plot_settings_action = self.menu_bar_action_handler_obj.create_configure_plot_settings_action()
        export_plot_action = self.menu_bar_action_handler_obj.create_export_plot_action()

        # arrange plot menu actions
        plot_menu_obj.addMenu(plot_submenu_obj)
        plot_menu_obj.addAction(config_plot_settings_action)
        plot_menu_obj.addSeparator()
        plot_menu_obj.addAction(export_plot_action)

        return plot_menu_obj


def load_license_text(file_path: str) -> str:
    """
    Reads the license text from the specified file path.

    :param file_path: Path to the LICENSE file.
    :return: Content of the LICENSE file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "LICENSE file not found."
    except Exception as e:
        return f"An error occurred while loading the LICENSE file: {e}"


class AboutDialog(qtw.QDialog):
    """
    About dialog with three tabs: About, License, and a blank tab for future use.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About FUSION Simulator")
        self.resize(600, 400)

        # Create tab widget
        tab_widget = qtw.QTabWidget(self)

        # About tab
        about_tab = qtw.QWidget()
        about_layout = qtw.QVBoxLayout(about_tab)
        about_text = qtw.QLabel(
            "FUSION Simulator\n\n"
            "Welcome to FUSION, an open-source venture into the future of networking.\n\n"
            "For more information, visit:"
        )
        about_text.setWordWrap(True)  # Enable word wrapping
        about_layout.addWidget(about_text)

        # Hyperlink
        link_label = qtw.QLabel(
            '<a href="https://github.com/SDNNetSim/FUSION/" style="color: blue;">'
            'https://github.com/SDNNetSim/FUSION/</a>'
        )
        link_label.setOpenExternalLinks(True)  # Make the link clickable
        about_layout.addWidget(link_label)

        tab_widget.addTab(about_tab, "About")

        # License tab
        license_tab = qtw.QWidget()
        license_layout = qtw.QVBoxLayout(license_tab)

        # Read the license text from the LICENSE file
        license_text = load_license_text("LICENSE")  # Adjust the file path as needed
        license_text_area = qtw.QPlainTextEdit(license_text)
        license_text_area.setReadOnly(True)  # Make it read-only
        license_layout.addWidget(license_text_area)

        tab_widget.addTab(license_tab, "License")

        # Blank tab for future
        blank_tab = qtw.QWidget()
        blank_layout = qtw.QVBoxLayout(blank_tab)
        placeholder_label = qtw.QLabel("Future functionality will go here.")
        blank_layout.addWidget(placeholder_label)
        tab_widget.addTab(blank_tab, "Blank Tab")

        # Main layout of the dialog
        main_layout = qtw.QVBoxLayout(self)
        main_layout.addWidget(tab_widget)

        # "Close" button
        close_button = qtw.QPushButton("Close")
        close_button.clicked.connect(self.close)
        main_layout.addWidget(close_button, alignment=qtc.Qt.AlignRight)


class MenuActionHandler:
    """
    Helper class to handle creation and functionality of menu bar actions.
    """

    def __init__(self, menu_bar: MenuBar, main_window_ref: qtw.QMainWindow):
        self.menu_bar_obj = menu_bar
        self.main_window_ref_obj = main_window_ref

    def create_load_config_from_file_action(self) -> qtw.QAction:
        """
        Create Load Config from File action

        :param : None
        :type : None
        :return: The action that was just created.
        :rtype: QtWidgets.QAction
        """
        load_config_action = qtw.QAction('Load Configuration from File', self.menu_bar_obj)
        load_config_action.triggered.connect(self._load_config_file)
        return load_config_action

    def create_display_topology_action(self) -> qtw.QAction:
        """
        Creates action to display a topology

        :param : None
        :type : None
        :return : Object representing the display topology action.
        :rtype : QWidgets.QAction
        """
        display_topology_action = qtw.QAction('Display topology', self.menu_bar_obj)
        display_topology_action.triggered.connect(self.display_topology)
        return display_topology_action

    def create_settings_action(self) -> qtw.QAction:
        """
        Creates the action to open the settings dialog

        :param : None
        :type : None
        :return : Object representing the settings action
        :rtype : QWidgets.QAction
        """
        settings_action = qtw.QAction('Settings', self.menu_bar_obj)
        settings_action.triggered.connect(self._open_settings)
        return settings_action

    def create_exit_action(self) -> qtw.QAction:
        """
        Creates action to stop application loop effectively quitting the GUI

        :param : None
        :type : None
        :return : Object representing the exit action.
        :rtype : QWidgets.QAction
        """
        exit_action = qtw.QAction('&Exit', self.menu_bar_obj)
        exit_action.triggered.connect(self.menu_bar_obj.app_close_signal_relay)
        return exit_action

    def create_plot_action(self) -> qtw.QAction:
        """
        Creates action for plotting

        :param : None
        :type : None
        :return : Object representing the plot action
        :rtype : QWidgets.QAction
        """
        plot_action = qtw.QAction("Plot", self.menu_bar_obj)
        plot_action.triggered.connect(self._plot_cb)
        return plot_action

    def create_configure_plot_settings_action(self) -> qtw.QAction:
        """
        Creates action for configuring plot settings

        :param : None
        :type : None
        :return : Object representing the configure plot settings action
        :rtype : QWidgets.QAction
        """
        configure_plot_settings = qtw.QAction("Configure Plot Settings", self.menu_bar_obj)
        configure_plot_settings.triggered.connect(self._configure_plot_settings_cb)
        return configure_plot_settings

    def create_export_plot_action(self) -> qtw.QAction:
        """
        Create action for exporting plots.

        :param : None
        :type : None
        :return : Object representing the export plot action
        :rtype : QWidgets.QAction
        """
        export_plot = qtw.QAction("Export Plot", self.menu_bar_obj)
        export_plot.triggered.connect(self._export_plot_cb)
        return export_plot

    def create_about_action(self) -> qtw.QAction:
        """
        Create action for displaying about section.

        :param : None
        :type : None
        :return : Object representing the "About" action
        :rtype : QtWidgets.QAction
        """
        about_action = qtw.QAction("About", self.menu_bar_obj)
        about_action.triggered.connect(self._show_about_dialog)       # connects to show dialog
        return about_action

    def _show_about_dialog(self):
        """
        Opens the custom About dialog with tabs.
        """
        about_dialog = AboutDialog(self.menu_bar_obj)
        about_dialog.exec()


    def _load_config_file(self):
        """
        Loads a configuration file from the user's system.

        :param : None
        :type : None
        :return : None
        :rtype : None
        """
        def get_config_file_name():
            """
            Prompts user to select the path of desired configuration file
            """
            # get any filename for opening
            file_name, _ = qtw.QFileDialog.getOpenFileName(
                parent=self.menu_bar_obj,
                caption="Load Configuration File",
                directory=qtc.QDir.homePath(),
                filter="INI files (*.ini)",
                options=qtw.QFileDialog.DontResolveSymlinks | qtw.QFileDialog.DontUseNativeDialog
            )
            return file_name

        config_file_name = get_config_file_name()
        print(f'{config_file_name}')

    def display_topology(self):
        """
        Displays a network topology.

        :param : None
        :type : None
        :return : None
        :rtype : None
        """
        def _display_topology(net_name):
            topo_information_dict, _ = create_network(net_name=net_name)

            edge_list = [(src, des, {'weight': link_len}) for (src, des), link_len in topo_information_dict.items()]
            network_topo = nx.Graph(edge_list)

            pos = nx.spring_layout(network_topo, seed=5, scale=2.0)  # Adjust the scale as needed

            # Create a canvas and plot the topology
            canvas = TopologyCanvas(self.main_window_ref_obj.mw_topology_view_area)
            canvas.plot(network_topo, pos)
            canvas.G = network_topo  # pylint: disable=invalid-name

            # Draw nodes using scatter to enable picking
            x, y = zip(*pos.values())  # pylint: disable=invalid-name
            scatter = canvas.axes.scatter(x, y, s=200)
            canvas.set_picker(scatter)

            self.main_window_ref_obj.mw_topology_view_area.setWidget(canvas)

        network_selection_dialog = qtw.QInputDialog()
        network_selection_dialog.setSizeGripEnabled(True)
        network_name, valid_net_name = network_selection_dialog.getItem(
            None, "Choose a network type:",
            "Select Network Type", GUI_DEFAULT_SETTINGS['supported_networks'], 0, False
        )

        if valid_net_name:
            _display_topology(net_name=network_name)

    @staticmethod
    def _open_settings():
        settings_dialog = SettingsDialog()
        if settings_dialog.exec() == qtw.QDialog.Accepted:
            print("Interaction complete")

    @staticmethod
    def _plot_cb():
        Alert(
            parent=None,
            title='Not Implemented',
            text='Plot Menu not implemented yet...come back soon!'
        ).alert(AlertCode.INFORMATION)

    @staticmethod
    def _configure_plot_settings_cb():
        Alert(
            parent=None,
            title='Not Implemented',
            text='Configure Plot settings menu not implemented yet...come back soon!'
        ).alert(AlertCode.INFORMATION)

    @staticmethod
    def _export_plot_cb():
        Alert(
            parent=None,
            title='Not Implemented',
            text='Export plot menu not implemented yet...come back soon!'
        ).alert(AlertCode.INFORMATION)

    @staticmethod
    def create_plot_sm_action(plot_action: str) -> qtw.QAction:
        """
        Creates a plot submenu action.

        :param plot_action : Name of the plot action
        :type plot_action : str
        :return : Object representing action associated with plot_action
        :rtype : QWidgets.QAction
        """
        def _user_alert_reason(pa):
            Alert(
                parent=None,
                title='Not Implemented',
                text=f'{pa} plot not implemented!'
            ).alert(AlertCode.INFORMATION)
        sm_action = qtw.QAction(f'Plot {plot_action}')
        sm_action.triggered.connect(lambda: _user_alert_reason(plot_action))
        return sm_action
