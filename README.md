@echo off
setlocal EnableDelayedExpansion

set "ThisScript=%~nx0"

:: Get current time in epoch minutes using PowerShell
for /f %%A in ('powershell -nologo -command "[int]((Get-Date).ToUniversalTime()-[datetime]::UnixEpoch).TotalMinutes"') do set Now=%%A

for /R %%F in (*) do (
    if /I not "%%~nxF"=="%ThisScript%" (
        for /f %%M in ('powershell -nologo -command "[int]((Get-Item \"%%~fF\").LastWriteTimeUtc - [datetime]::UnixEpoch).TotalMinutes"') do (
            set /a Age=%Now%-%%M
            if !Age! GEQ 60 del /f /q "%%~fF" >nul 2>&1
        )
    )
)

for /F "delims=" %%D in ('dir /ad/b/s /o-n') do rd "%%D" 2>nul

endlocal
exit /b