@echo off
:: Deletes files and folders older than 1 hour in the current directory

setlocal enabledelayedexpansion

:: Set reference time (1 hour ago)
for /f "tokens=1-2 delims==" %%i in ('"wmic os get localdatetime /value"') do if /i "%%i"=="LocalDateTime" set ldt=%%j
set currentTime=%ldt:~8,6%
set currentDate=%ldt:~0,8%

echo Current DateTime: %currentDate% %currentTime%

:: Create a temporary list of files older than 1 hour
forfiles /P "%cd%" /S /D -0 /C "cmd /c if @isdir==FALSE call :checktime @path"

:: Done
echo Cleanup complete.
goto :eof

:checktime
set file=%~1
for %%T in ("%file%") do (
    set modtime=%%~tT
)
:: Convert timestamps to minutes
for /f "tokens=1,2 delims= " %%a in ("!modtime!") do (
    set fDate=%%a
    set fTime=%%b
)

:: Compare with current time (using PowerShell)
for /f %%x in ('powershell -NoProfile -Command "(Get-Date '%fDate% %fTime%') -lt (Get-Date).AddHours(-1)"') do set result=%%x
if /i "!result!"=="True" (
    echo Deleting: %file%
    del /f /q "%file%" >nul 2>&1
    rd /s /q "%file%" >nul 2>&1
)
goto :eof