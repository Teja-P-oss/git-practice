@echo off
setlocal enabledelayedexpansion

REM Set the device directory and destination directory on your PC
set "REMOTE_DIR=/data/vendor/camena"
set "DEST_DIR=C:\Users\teja.potti.SECDS\Desktop"

REM Initialize an array to keep track of pulled files to avoid duplicates
set "PULLED_FILES="

REM First, list files matching *Reprocess*.yuv and *Reprocess*.raw
for /f "delims=" %%F in ('adb shell "find %REMOTE_DIR% -type f \( -name ""*Reprocess*.yuv"" -o -name ""*Reprocess*.raw"" \) 2>/dev/null"') do (
    set "FILE=%%F"
    REM Extract the filename from the full path
    for %%I in ("!FILE!") do set "FILENAME=%%~nxI"
    
    REM Check if the filename contains more than one of the patterns: MCFP, RGBP, BYRP
    set "COUNT=0"
    for %%P in (MCFP RGBP BYRP) do (
        echo "!FILENAME!" | findstr /i "%%P" >nul && (
            set /a COUNT+=1
        )
    )
    if !COUNT! gtr 1 (
        REM Check if the file has already been pulled
        echo "!PULLED_FILES!" | findstr /i "!FILENAME!" >nul || (
            REM Pull the file
            adb pull "!FILE!" "%DEST_DIR%"
            REM Add the filename to the list of pulled files
            set "PULLED_FILES=!PULLED_FILES! !FILENAME!"
        )
    )
)

REM Apart from the above, if an LME file is present once, pull it once
REM We need to ensure we don't pull an LME file if it's already been pulled
set "LME_PULLED=0"
for /f "delims=" %%F in ('adb shell "find %REMOTE_DIR% -type f -name ""*LME*"" 2>/dev/null"') do (
    set "FILE=%%F"
    for %%I in ("!FILE!") do set "FILENAME=%%~nxI"
    REM Check if we've already pulled this file
    echo "!PULLED_FILES!" | findstr /i "!FILENAME!" >nul || (
        if !LME_PULLED! equ 0 (
            adb pull "!FILE!" "%DEST_DIR%"
            set "LME_PULLED=1"
            REM Optionally, add it to the list of pulled files
            set "PULLED_FILES=!PULLED_FILES! !FILENAME!"
        )
    )
    if !LME_PULLED! equ 1 goto :EndLME
)
:EndLME

endlocal
pause
