# Запуск для жюри: останавливает старые копии, открывает окна «сервис» и «туннель», ждёт готовности и показывает публичный адрес.
Set-Location (Split-Path $PSScriptRoot -Parent)
Write-Host "1/4 Останавливаю старые копии сервиса и туннеля..."
$l = (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue).OwningProcess | Select-Object -First 1
if ($l) {
    $par = (Get-CimInstance Win32_Process -Filter "ProcessId=$l").ParentProcessId
    $top = Get-CimInstance Win32_Process -Filter "ProcessId=$par"
    taskkill /PID $(if ($top.CommandLine -match 'uvicorn|serve.py') { $par } else { $l }) /T /F | Out-Null
}
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'tools[\\/]serve\.py|run_service\.ps1|run_tunnel\.ps1' -and $_.ProcessId -ne $PID } | ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null | Out-Null }
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item tools\PUBLIC_URL.txt, tools\tunnel.log -ErrorAction SilentlyContinue
Start-Sleep 2

Write-Host "2/4 Запускаю сервис (окно «Forte · сервис»)..."
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$PSScriptRoot\run_service.ps1`""
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep 3
    try { $h = Invoke-RestMethod http://127.0.0.1:8765/health -TimeoutSec 3; if ($h.ready_workers -eq 5) { $ok = $true; break } } catch {}
}
if (-not $ok) { Write-Host "Сервис не поднялся за 3 минуты. Смотрите tools\serve.log" -ForegroundColor Red; exit 1 }
Write-Host "   сервис готов: 5 воркеров" -ForegroundColor Green

Write-Host "3/4 Запускаю туннель (окно «Forte · туннель»)..."
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$PSScriptRoot\run_tunnel.ps1`""
for ($i = 0; $i -lt 40 -and -not (Test-Path tools\PUBLIC_URL.txt); $i++) { Start-Sleep 2 }
if (-not (Test-Path tools\PUBLIC_URL.txt)) { Write-Host "Туннель не выдал адрес. Смотрите окно туннеля." -ForegroundColor Red; exit 1 }
$url = (Get-Content tools\PUBLIC_URL.txt).Trim()

Write-Host "4/4 Проверяю публичный адрес..."
$pub = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep 3
    try { $h = Invoke-RestMethod "$url/health" -TimeoutSec 10; if ($h.ready_workers -eq 5) { $pub = $true; break } } catch {}
}
Write-Host ""
if ($pub) { Write-Host "ГОТОВО. Публичный адрес отвечает." -ForegroundColor Green } else { Write-Host "Адрес получен, но пока не отвечает (DNS Cloudflare может обновляться до минуты). Проверьте позже: $url/health" -ForegroundColor Yellow }
Write-Host ""
Write-Host "  Web interface URL : $url"
Write-Host "  API base URL      : $url/api/v1"
Write-Host "  API key           : SERVICE_API_KEY из service\.env"
Set-Clipboard -Value $url
Write-Host ""
Write-Host "Адрес скопирован в буфер. Окна «Forte · сервис» и «Forte · туннель» НЕ ЗАКРЫВАТЬ до 14:30."
