Param(
    [string]$LocalFile = "release\dataset_all.zip",
    [string]$FtpHost = "ftp.example.com",
    [string]$RemotePath = "/upload/dataset_all.zip"
)

if (-not (Test-Path $LocalFile)) { Write-Error "Local file $LocalFile not found"; exit 1 }

$user = $env:FTP_USER
$pass = $env:FTP_PASS
if (-not $user -or -not $pass) { Write-Error "Set FTP_USER and FTP_PASS in environment"; exit 1 }

$uri = "ftp://$FtpHost$RemotePath"
[System.Net.ServicePointManager]::ServerCertificateValidationCallback = {$true}
$webclient = New-Object System.Net.WebClient
$webclient.Credentials = New-Object System.Net.NetworkCredential($user,$pass)
$webclient.UploadFile($uri, $LocalFile)
Write-Host "Uploaded $LocalFile to $uri"
