@echo off
setlocal EnableDelayedExpansion

:: Get current script name
set "ThisScript=%~nx0"

:: Get current time as reference (in minutes since epoch)
for /f "tokens=2 delims==." %%A in ('"wmic os get localdatetime /value"') do set datetime=%%A
set "YYYY=%datetime:~0,4%"
set "MM=%datetime:~4,2%"
set "DD=%datetime:~6,2%"
set "HH=%datetime:~8,2%"
set "Min=%datetime:~10,2%"

:: Convert current time to minutes since epoch
set /a "NowMinutes = (((%YYYY%-1970)*365 + (%MM%-1)*30 + %DD%) * 24 + %HH%) * 60 + %Min%"

:: Loop through files
for /R %%F in (*) do (
    if /I not "%%~nxF"=="%ThisScript%" (
        call :CheckAndDelete "%%F"
    )
)

:: Loop through folders (bottom-up)
for /F "delims=" %%D in ('dir /ad/b/s /o-n') do (
    rd "%%D" 2>nul
)

exit /b

:CheckAndDelete
set "target=%~1"
:: Get modified date/time of the file
for %%T in ("%target%") do (
    set "File=%%~fT"
    set "ModDate=%%~tT"
)

:: Parse date and time from ModDate
for /f "tokens=1,2 delims= " %%A in ("!ModDate!") do (
    set "FDate=%%A"
    set "FTime=%%B"
)

:: Extract components
for /f "tokens=1-3 delims=/-" %%A in ("!FDate!") do (
    set "MM=%%A"
    set "DD=%%B"
    set "YYYY=%%C"
)
for /f "tokens=1,2 delims=:" %%H in ("!FTime!") do (
    set "HH=%%H"
    set "Min=%%I"
)

:: Convert to 24-hour time if PM is present
echo !FTime! | find "PM" >nul
if !errorlevel! == 0 if not "!HH!"=="12" set /a HH+=12
echo !FTime! | find "AM" >nul
if !errorlevel! == 0 if "!HH!"=="12" set "HH=00"

:: Remove any AM/PM suffix
set "Min=!Min:AM=!"
set "Min=!Min:PM=!"

:: Convert file modified time to minutes since epoch
set /a "FileMinutes = (((!YYYY!-1970)*365 + (!MM!-1)*30 + !DD!) * 24 + !HH!) * 60 + !Min!"

:: Compare difference
set /a "Age=!NowMinutes! - !FileMinutes!"

if !Age! GEQ 60 (
    del /f /q "!target!" >nul 2>&1
)

exit /b