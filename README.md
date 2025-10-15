@echo off
setlocal

rem === EraseJunk_1hour_Limit.bat ===
rem Recursively deletes files older than 1 hour under the current directory,
rem then removes empty subfolders older than 1 hour.
rem Operates ONLY within the current working directory and skips this .bat file.

set "SELFPATH=%~f0"

echo Cleaning "%CD%" (recursively) for items older than 1 hour...

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $limit=(Get-Date).AddHours(-1); $selfPath=[System.IO.Path]::GetFullPath('%SELFPATH%'); $files=Get-ChildItem -LiteralPath '.' -Recurse -Force -File | Where-Object { $_.FullName -ne $selfPath -and $_.LastWriteTime -lt $limit }; $countFiles=$files.Count; if($countFiles -gt 0){ $files | Remove-Item -Force }; $dirs=Get-ChildItem -LiteralPath '.' -Recurse -Force -Directory | Sort-Object FullName -Descending; $removedDirs=0; foreach($d in $dirs){ if(-not (Get-ChildItem -LiteralPath $d.FullName -Force -Recurse)){ if($d.LastWriteTime -lt $limit){ Remove-Item -LiteralPath $d.FullName -Force; $removedDirs++ } } }; Write-Host ('Removed ' + $countFiles + ' file(s) and ' + $removedDirs + ' directorie(s).');"
endlocal