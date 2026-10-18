# Окно «Forte · туннель»: публичный адрес Cloudflare. При обрыве перезапускается (адрес при этом меняется — см. tools\PUBLIC_URL.txt).
$Host.UI.RawUI.WindowTitle = "Forte · туннель (не закрывать до 14:30)"
Set-Location (Split-Path $PSScriptRoot -Parent)
while ($true) {
    Write-Host "$(Get-Date -Format HH:mm:ss) запуск туннеля..."
    & .\tools\cloudflared\cloudflared.exe tunnel --no-autoupdate --url http://127.0.0.1:8765 2>&1 | ForEach-Object {
        $line = "$_"; Add-Content -Path tools\tunnel.log -Value $line
        if ($line -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
            $Matches[0] | Set-Content -Path tools\PUBLIC_URL.txt
            Write-Host "ПУБЛИЧНЫЙ АДРЕС: $($Matches[0])" -ForegroundColor Green
        }
    }
    Write-Host "$(Get-Date -Format HH:mm:ss) туннель оборвался — перезапуск через 5 с (АДРЕС ИЗМЕНИТСЯ)" -ForegroundColor Yellow
    Start-Sleep 5
}
