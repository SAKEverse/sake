:: Check if Miniconda is present
IF EXIST %USERPROFILE%\miniconda3\Scripts\activate.bat (
    set conda_file=%USERPROFILE%\miniconda3
    GOTO FOUND
)

IF EXIST %USERPROFILE%\AppData\Local\miniconda3\Scripts\activate.bat (
    set conda_file=%USERPROFILE%\AppData\Local\miniconda3
    GOTO FOUND
)

IF EXIST %USERPROFILE%\AppData\Local\anaconda3\Scripts\activate.bat (
    set conda_file=%USERPROFILE%\AppData\Local\anaconda3
    GOTO FOUND
)

IF EXIST %USERPROFILE%\anaconda3\Scripts\activate.bat (
    set conda_file=%USERPROFILE%\anaconda3
    GOTO FOUND
)

echo Miniconda or Anaconda not found.
pause
exit /b

:FOUND

:: enter conda
call %conda_file%\Scripts\activate.bat

:: Navigate sake-plan directory
cd %USERPROFILE%\Documents\GitHub\sake

:: Check if environment exists
IF NOT EXIST %conda_file%\envs\sake\python.exe (
    conda env create -f environment.yml
)

:: Activate conda environment
call activate sake

:: Launch app
cd sakeplot
python sakeplot.py

TIMEOUT 10