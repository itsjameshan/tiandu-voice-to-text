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
foreach ($item in @("app.py", "config.yaml", "pipeline", "tools", "data", "docs", "reports", "README.md", "GLOSSARY.md")) {
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
$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:PYTHONUTF8 = "1"
& (Join-Path $PyDir "python.exe") "tools\selfcheck.py"
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "自检没有通过" }

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
