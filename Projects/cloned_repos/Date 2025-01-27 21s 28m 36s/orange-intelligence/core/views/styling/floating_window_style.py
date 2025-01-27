class FloatingWindowStyleOptions:
    geometry = "200, 200, 400, 300"

    title = "Orange GUI"

    base = """
        QWidget {
            background-color: #f4f4f4; 
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 0;
        }
    """

    tab_widget = """
        QTabWidget::pane { 
            border: none; 
            background: transparent; 
        }
        QTabBar::tab { 
            background: #f8f8f8; 
            border: none; 
            padding: 8px 16px; 
            margin: 2px; 
            border-top-left-radius: 8px; 
            border-top-right-radius: 8px; 
        }
        QTabBar::tab:selected { 
            background: #ffffff; 
            font-weight: bold; 
            color: #007aff; 
        }
        QTabBar::tab:hover { 
            background: #e9e9e9; 
        }
    """

    search_bar = """
        QLineEdit { 
            background-color: #ffffff; 
            border: 1px solid #d1d1d1; 
            border-radius: 6px; 
            padding: 6px; 
            margin: 6px 0; 
        }
    """

    list_widget = """
        QListWidget { 
            background-color: #ffffff; 
            border: 1px solid #d1d1d1; 
            border-radius: 6px; 
            margin: 8px; 
        }
        QListWidget::item { 
            padding: 8px; 
            font-size: 14px; 
            color: #333; 
        }
        QListWidget::item:selected { 
            background-color: #007aff; 
            color: #ffffff; 
            font-weight: bold; 
            border-radius: 4px; 
            margin: 2px; 
        }
        QListWidget::item:hover { 
            background-color: #e0e0e0; 
            color: #000; 
        }
    """
