@echo off
:: Deletes files and folders older than 1 hour in the current directory (excluding this script)

setlocal enabledelayedexpansion

:: --- Get current date and time ---
for /f "tokens=1-2 delims==" %%i in ('"wmic os get localdatetime /value"') do if /i "%%i"=="LocalDateTime" set ldt=%%j
set currentDate=%ldt:~0,8%
set currentTime=%ldt:~8,6%
set scriptPath=%~f0

echo.
echo =============================================
echo Cleaning directory: %cd%
echo Current time: %currentDate% %currentTime%
echo Script file:  %scriptPath%
echo =============================================
echo.

:: --- Check each file/folder in current directory recursively ---
for /r "%cd%" %%F in (*) do call :checktime "%%F"

echo.
echo Cleanup complete.
goto :eof

:: --- Subroutine to check age and delete if older than 1 hour ---
:checktime
set file=%~1

:: Skip this script itself
if /i "%file%"=="%scriptPath%" (
    echo Skipping script file: %file%
    goto :eof
)

:: Get file modification time
for %%T in ("%file%") do set modtime=%%~tT
for /f "tokens=1,2 delims= " %%a in ("!modtime!") do (
    set fDate=%%a
    set fTime=%%b
)

:: Use PowerShell to check if file is older than 1 hour
for /f %%x in ('powershell -NoProfile -Command "(Get-Date '%fDate% %fTime%') -lt (Get-Date).AddHours(-1)"') do set result=%%x

if /i "!result!"=="True" (
    echo Deleting: %file%
    del /f /q "%file%" >nul 2>&1
    rd /s /q "%file%" >nul 2>&1
)
goto :eof