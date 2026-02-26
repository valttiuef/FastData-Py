from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QApplication, QMessageBox
from .localization import tr



def init_menu_bar(window):
    menubar = window.menuBar()

    file_menu = menubar.addMenu(tr("File"))

    new_action = QAction(tr("New Database..."), window)
    new_action.setShortcut(QKeySequence("Ctrl+N"))
    file_menu.addAction(new_action)
    new_action.triggered.connect(lambda: window.new_database())

    open_action = QAction(tr("Open Database..."), window)
    open_action.setShortcut(QKeySequence("Ctrl+O"))
    file_menu.addAction(open_action)
    open_action.triggered.connect(lambda: window.open_database())

    save_as_action = QAction(tr("Save Database..."), window)
    save_as_action.setShortcut(QKeySequence("Ctrl+S"))
    save_as_action.triggered.connect(lambda: window.save_database_as())
    file_menu.addAction(save_as_action)

    file_menu.addSeparator()

    reset_db_action = QAction(tr("Reset Database..."), window)
    reset_db_action.triggered.connect(lambda: _confirm_reset_database(window))
    file_menu.addAction(reset_db_action)

    refresh_action = QAction(tr("Refresh Databases"), window)
    refresh_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
    refresh_action.triggered.connect(lambda: _confirm_refresh_databases(window))
    file_menu.addAction(refresh_action)

    file_menu.addSeparator()

    quit_action = QAction(tr("Quit"), window)
    quit_action.setShortcut(QKeySequence("Ctrl+Q"))
    quit_action.triggered.connect(lambda: QApplication.quit())
    file_menu.addAction(quit_action)

    view_menu = menubar.addMenu(tr("View"))

    log_action = QAction(tr("Log"), window, checkable=True)
    log_action.setShortcut(QKeySequence("Ctrl+Shift+G"))
    log_action.toggled.connect(window.set_log_visible)
    view_menu.addAction(log_action)

    chat_action = QAction(tr("Chat"), window)
    chat_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
    chat_action.triggered.connect(window.show_chat_window)
    view_menu.addAction(chat_action)

    view_menu.addSeparator()

    language_menu = view_menu.addMenu(tr("Language"))
    language_group = QActionGroup(window)
    language_group.setExclusive(True)
    window.language_actions = {}
    for code, label in window.available_languages().items():
        action = QAction(label, window, checkable=True)
        action.triggered.connect(lambda checked=False, c=code: window.change_language(c))
        language_group.addAction(action)
        language_menu.addAction(action)
        window.language_actions[code] = action

    theme_menu = view_menu.addMenu(tr("Theme"))

    group = QActionGroup(window)
    group.setExclusive(True)

    action_light = QAction(tr("Light"), window, checkable=True)
    action_dark = QAction(tr("Dark"), window, checkable=True)

    group.addAction(action_light)
    group.addAction(action_dark)
    theme_menu.addAction(action_light)
    theme_menu.addAction(action_dark)

    action_light.setShortcut(QKeySequence("Ctrl+Shift+L"))
    action_dark.setShortcut(QKeySequence("Ctrl+Shift+D"))

    action_light.triggered.connect(lambda: window.apply_theme("light", apply_runtime=False))
    action_dark.triggered.connect(lambda: window.apply_theme("dark", apply_runtime=False))

    help_menu = menubar.addMenu(tr("Help"))
    about_action = QAction(tr("About"), window)
    about_action.triggered.connect(window.show_about_dialog)
    help_menu.addAction(about_action)
    docs_action = QAction(tr("Documentation"), window)
    docs_action.triggered.connect(window.open_help_docs)
    help_menu.addAction(docs_action)

    window.action_light = action_light
    window.action_dark = action_dark
    window.action_show_log = log_action
    window.action_open_chat = chat_action

    return menubar


def _confirm_reset_database(window):
    reply = QMessageBox.warning(
        window,
        tr("Reset Database"),
        tr("Are you sure you want to reset the database?\n\nThis will clear all data and restore the database to its default empty state. This cannot be undone."),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.Yes:
        window.reset_database()


def _confirm_refresh_databases(window):
    reply = QMessageBox.information(
        window,
        tr("Refresh Databases"),
        tr("Refresh will reload all database files from disk.\n\nThis may take some time depending on the size of your databases.\n\nDo you want to continue?"),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.Yes:
        window.refresh_databases()
