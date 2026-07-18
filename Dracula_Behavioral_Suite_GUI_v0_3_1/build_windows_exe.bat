@echo off
py -m pip install -r requirements.txt
py -m PyInstaller --noconfirm --clean --windowed ^
  --name DraculaBehavioralSuite ^
  --collect-all PySide6 ^
  --collect-all cv2 ^
  --collect-all matplotlib ^
  --collect-all openpyxl ^
  app.py
echo.
echo Build complete.
echo Check dist\DraculaBehavioralSuite\
pause
