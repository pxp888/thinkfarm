import sys
from PyQt6.QtWidgets import QApplication
from app_gui import ThinkfarmApp

def main():
    app = QApplication(sys.argv)
    window = ThinkfarmApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
