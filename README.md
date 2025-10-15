@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$cutoff = (Get-Date).AddHours(-1); ^
$scriptPath = Join-Path (Get-Location) 'EraseJunk_1hour_Limit.bat'; ^
Get-ChildItem -Path . -Recurse -File | Where-Object { $_.LastWriteTime -lt $cutoff -and $_.FullName -ne $scriptPath } | Remove-Item -Force; ^
$dirs = Get-ChildItem -Path . -Recurse -Directory | Sort-Object { [regex]::Matches($_.FullName, '\\\\').Count } -Descending; ^
foreach ($dir in $dirs) { ^
    if ($dir.LastWriteTime -lt $cutoff -and -not (Get-ChildItem -LiteralPath $dir.FullName -Force)) { ^
        Remove-Item -LiteralPath $dir.FullName -Force ^
    } ^
}"