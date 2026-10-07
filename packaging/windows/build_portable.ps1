# 构建 Windows 免安装便携包（在 GitHub Actions 的 windows-latest 上运行，也可以在一台联网的 Windows 电脑上运行）
# 用法：pwsh packaging/windows/build_portable.ps1 -Version v0.1.0
# 前提：已安装与 -PythonVersion 同一小版本（3.11）的 Python，并在 PATH 里叫 python。
param(
    [string]$PythonVersion = "3.11.9",
    [string]$OutDir = "dist",
    [string]$Version = "dev"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # 关掉进度条，下载快很多
$env:PYTHONUTF8 = "1"                      # Python 输出中文时用 UTF-8，避免打印报错
$env:PYTHONIOENCODING = "utf-8"
$env:GRADIO_ANALYTICS_ENABLED = "False"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Dist = Join-Path $Root $OutDir
$Pkg = Join-Path $Dist "tiandu"
$PyDir = Join-Path $Pkg "python"
if (Test-Path $Pkg) { Remove-Item -Recurse -Force $Pkg }
New-Item -ItemType Directory -Force -Path $PyDir | Out-Null

Write-Host "== 1. 下载 Python $PythonVersion 嵌入版"
$embedZip = Join-Path $Dist "python-$PythonVersion-embed-amd64.zip"
if (-not (Test-Path $embedZip)) {
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $embedZip
}
Expand-Archive -Path $embedZip -DestinationPath $PyDir -Force

Write-Host "== 2. 设置搜索路径（._pth）"
$parts = $PythonVersion.Split(".")
$short = "$($parts[0])$($parts[1])"
$pth = Join-Path $PyDir "python$short._pth"
Set-Content -Path $pth -Encoding ascii -Value @("python$short.zip", ".", "Lib\site-packages", "..", "import site")

Write-Host "== 3. 安装依赖到便携 Python"
$site = Join-Path $PyDir "Lib\site-packages"
python -m pip install --disable-pip-version-check --no-warn-script-location --target $site -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "安装依赖失败" }

Write-Host "== 4. 复制项目文件"
foreach ($item in @("app.py", "config.yaml", "pipeline", "tools", "data", "docs", "reports", "README.md", "GLOSSARY.md",
                       "requirements.txt", "requirements-tf.txt")) {
    $src = Join-Path $Root $item
    if (Test-Path $src) { Copy-Item -Recurse -Force $src (Join-Path $Pkg $item) }
}
New-Item -ItemType Directory -Force -Path (Join-Path $Pkg "models") | Out-Null
Copy-Item (Join-Path $Root "models\download_models.py") (Join-Path $Pkg "models\download_models.py")
foreach ($f in @("start.bat", "selfcheck.bat", "README.txt")) {
    Copy-Item (Join-Path $PSScriptRoot $f) (Join-Path $Pkg $f)
}

Write-Host "== 5. 下载模型"
& (Join-Path $PyDir "python.exe") (Join-Path $Pkg "models\download_models.py")
if ($LASTEXITCODE -ne 0) { throw "下载模型失败" }

Write-Host "== 6. 自检（用便携 Python 处理测试音频）"
Push-Location $Pkg
& (Join-Path $PyDir "python.exe") "tools\selfcheck.py"
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "自检没有通过" }

Write-Host "== 6b. 启动网页工具，确认能打开（用便携 Python）"
$proc = Start-Process -FilePath (Join-Path $PyDir "python.exe") -ArgumentList @("app.py", "--host", "127.0.0.1", "--port", "7861") `
    -WorkingDirectory $Pkg -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $Dist "app_stdout.log") `
    -RedirectStandardError (Join-Path $Dist "app_stderr.log")
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:7861/" -UseBasicParsing -TimeoutSec 5
        if ($resp.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    if ($proc.HasExited) { break }
}
if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
if (-not $ok) {
    Get-Content (Join-Path $Dist "app_stderr.log") -ErrorAction SilentlyContinue | Select-Object -Last 40
    throw "网页工具没有启动成功"
}
Write-Host "网页工具能正常打开"

Write-Host "== 7. 清理并打包"
foreach ($d in @("outputs", "tmp", "data_pool")) {
    $p = Join-Path $Pkg $d
    if (Test-Path $p) { Remove-Item -Recurse -Force $p }
}
Get-ChildItem -Path $Pkg -Recurse -Directory -Filter "__pycache__" | Where-Object { $_.FullName -notlike "*\python\*" } | Remove-Item -Recurse -Force
$zip = Join-Path $Dist "tiandu-portable-win64-$Version.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $Pkg -DestinationPath $zip -CompressionLevel Optimal
$sizeMB = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "完成：$zip（$sizeMB MB）"
