# Окно «Forte · сервис»: держит сервис запущенным (перезапуск при падении) и не даёт Windows уснуть, пока окно открыто.
$Host.UI.RawUI.WindowTitle = "Forte · сервис (не закрывать до 14:30)"
Set-Location (Split-Path $PSScriptRoot -Parent)
Add-Type -Namespace Forte -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Forte.Power]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: запрет сна без изменения настроек Windows
Remove-Item Env:SERVICE_API_KEY -ErrorAction SilentlyContinue
while ($true) {
    Write-Host "$(Get-Date -Format HH:mm:ss) запуск сервиса..."
    & .\service\.venv\Scripts\python.exe tools\serve.py *>> tools\serve.log
    Write-Host "$(Get-Date -Format HH:mm:ss) сервис остановился — перезапуск через 5 с" -ForegroundColor Yellow
    Start-Sleep 5
}
