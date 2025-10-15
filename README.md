@echo off
setlocal EnableDelayedExpansion
set "ThisScript=%~nx0"
for /f "tokens=2 delims==." %%A in ('"wmic os get localdatetime /value"') do set t=%%A
set /a Now=(((((%t:~0,4%-1970)*365+(%t:~4,2%-1)*30+%t:~6,2%)*24+%t:~8,2%)*60)+%t:~10,2%)

for /R %%F in (*) do (
    if /I not "%%~nxF"=="%ThisScript%" (
        for %%T in ("%%F") do set "f=%%~fT" & set "d=%%~tT"
        for /f "tokens=1,2 delims= " %%a in ("!d!") do (
            for /f "tokens=1-3 delims=/-" %%d in ("%%a") do (
                for /f "tokens=1,2 delims=:" %%h in ("%%b") do (
                    set "y=%%f" & set "m=%%d" & set "dy=%%e" & set "hh=%%h" & set "mi=%%i"
                )
            )
        )
        echo !d! | find "PM" >nul && if not "!hh!"=="12" set /a hh+=12
        echo !d! | find "AM" >nul && if "!hh!"=="12" set hh=00
        set "mi=!mi:AM=!" & set "mi=!mi:PM=!"
        set /a F=(((!y!-1970)*365+(!m!-1)*30+!dy!)*24+!hh!)*60+!mi!
        set /a A=!Now!-!F!
        if !A! GEQ 60 del /f /q "%%F" >nul 2>&1
    )
)

for /F "delims=" %%D in ('dir /ad/b/s /o-n') do rd "%%D" 2>nul
exit /b