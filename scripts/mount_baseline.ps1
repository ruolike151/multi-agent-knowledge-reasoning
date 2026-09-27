# 把参考项目以目录联接（junction）方式挂到仓库的 baseline/ 下，
# 只读参考，不复制源码（设计决策 D-02）。
#
# 用法：先自行下载并解压参考项目（https://github.com/bcefghj/agent-knowledge-hub），
# 再把解压目录传给本脚本：
#   .\scripts\mount_baseline.ps1 -Source <解压后的目录>
param(
    [Parameter(Mandatory = $true)]
    [string]$Source
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$target = Join-Path $repo "baseline"
$source = $Source

if (-not (Test-Path -LiteralPath $source)) { throw "参考项目不存在：$source" }
$source = (Resolve-Path -LiteralPath $source).Path
if (Test-Path -LiteralPath $target) { throw "挂载点已存在：$target（如需重建请先删除该联接）" }

New-Item -ItemType Junction -Path $target -Target $source | Out-Null
Write-Output "已挂载：$target -> $source"
