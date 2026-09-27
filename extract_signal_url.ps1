# ==============================================================================
# Zenless Zone Zero (ZZZ) - Signal History Authkey Extractor
# Projek Penambangan Data Semester 5
# Fitur: Auto-detect path, Bypass file-lock, Validasi status authkey
# ==============================================================================

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$ProgressPreference = "SilentlyContinue"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   ZZZ SIGNAL HISTORY EXTRACTOR - DATA MINING PROJECT     " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

$candidate_dirs = @()

# 1. Cek dari Process ZZZ yang sedang berjalan
$proc = Get-Process ZenlessZoneZero -ErrorAction SilentlyContinue
if ($proc -and $proc.Path) {
    $procDir = Split-Path -Parent $proc.Path
    $candidate_dirs += "$procDir\ZenlessZoneZero_Data"
}

# 2. Cek dari Windows Registry (Steam / HoYoPlay)
$regPaths = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
)
foreach ($reg in $regPaths) {
    $items = Get-ItemProperty $reg -ErrorAction SilentlyContinue | Where-Object { 
        $_.DisplayName -like "*Zenless Zone Zero*" -or $_.InstallPath -like "*Zenless*" 
    }
    foreach ($item in $items) {
        if ($item.InstallLocation) {
            $candidate_dirs += "$($item.InstallLocation)\games\ZenlessZoneZero Game\ZenlessZoneZero_Data"
            $candidate_dirs += "$($item.InstallLocation)\ZenlessZoneZero_Data"
        }
        if ($item.InstallPath) {
            $candidate_dirs += "$($item.InstallPath)\games\ZenlessZoneZero Game\ZenlessZoneZero_Data"
            $candidate_dirs += "$($item.InstallPath)\ZenlessZoneZero_Data"
        }
    }
}

# 3. Cek dari Unity Player.log
$locallow = "$env:USERPROFILE\AppData\LocalLow\miHoYo\ZenlessZoneZero\Player.log"
if (Test-Path $locallow) {
    $logLines = Get-Content $locallow -First 20 -ErrorAction SilentlyContinue
    $prefix = "[Subsystems] Discovering subsystems at path "
    $suffix = "/UnitySubsystems"
    foreach ($line in $logLines) {
        if ($line.StartsWith($prefix)) {
            $parsed = $line.Substring($prefix.Length, $line.Length - $prefix.Length - $suffix.Length)
            $candidate_dirs += $parsed
            break
        }
    }
}

# 4. Fallback Path Umum (Steam Library di berbagai drive)
$drives = Get-PSDrive -PSProvider FileSystem | Select-Object -ExpandProperty Root
foreach ($d in $drives) {
    $candidate_dirs += "$($d)steam\steamapps\common\Zenless Zone Zero\games\ZenlessZoneZero Game\ZenlessZoneZero_Data"
    $candidate_dirs += "$($d)SteamLibrary\steamapps\common\Zenless Zone Zero\games\ZenlessZoneZero Game\ZenlessZoneZero_Data"
    $candidate_dirs += "$($d)Program Files\ZenlessZoneZero Game\ZenlessZoneZero_Data"
}

# Bersihkan dan filter direktori yang valid
$valid_dirs = $candidate_dirs | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

if ($valid_dirs.Count -eq 0) {
    Write-Host "[!] Gagal menemukan folder instalasi Zenless Zone Zero secara otomatis." -ForegroundColor Red
    Write-Host "    Pastikan game ZZZ sudah pernah diinstall di komputer ini." -ForegroundColor Yellow
    exit 1
}

# Cari file data_2 pada semua versi webCaches
$cache_files = @()
foreach ($dir in $valid_dirs) {
    $webCaches = "$dir\webCaches"
    if (Test-Path $webCaches) {
        $files = Get-ChildItem -Path "$webCaches\*\Cache\Cache_Data\data_2" -ErrorAction SilentlyContinue
        if ($files) {
            $cache_files += $files
        }
    }
}

if ($cache_files.Count -eq 0) {
    Write-Host "[!] File cache gacha (data_2) belum ditemukan." -ForegroundColor Red
    Write-Host "    Silakan buka game ZZZ, masuk ke menu Signal Search (Gacha), dan buka halaman 'History'." -ForegroundColor Yellow
    exit 1
}

# Urutkan berdasarkan waktu modifikasi terbaru
$cache_files = $cache_files | Sort-Object LastWriteTime -Descending

Write-Host "[*] Memeriksa file cache gacha terbaru..." -ForegroundColor Cyan

$found_url = $null
$is_expired = $false

foreach ($file in $cache_files) {
    $tempFile = [IO.Path]::GetTempFileName()
    try {
        # Copy file cache ke TEMP agar tidak konflik lock jika game sedang berjalan
        Copy-Item -LiteralPath $file.FullName -Destination $tempFile -Force
        $content = [IO.File]::ReadAllText($tempFile, [System.Text.Encoding]::GetEncoding("latin1"))
        
        $chunks = $content -split "1/0/"
        foreach ($chunk in $chunks) {
            if ($chunk.StartsWith("http") -and $chunk.Contains("getGachaLog") -and $chunk.Contains("authkey=")) {
                $match = [regex]::Match($chunk, 'https?://[^\x00-\x20\x7F-\xFF]+')
                if ($match.Success) {
                    $testUrl = ($match.Value -split "&end_id=")[0] + "&end_id=0&size=5"
                    
                    # Validasi apakah token masih aktif atau sudah expired
                    try {
                        $res = Invoke-RestMethod -Uri $testUrl -Method Get -TimeoutSec 5 -ErrorAction Stop
                        if ($res.retcode -eq 0) {
                            $found_url = $testUrl
                            break
                        } elseif ($res.retcode -eq -1 -and $res.message -like "*time out*") {
                            $is_expired = $true
                        }
                    } catch {
                        # Abaikan network error sesaat, lanjut cek baris berikutnya
                    }
                }
            }
        }
    } finally {
        if (Test-Path $tempFile) {
            Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue
        }
    }

    if ($found_url) {
        break
    }
}

Write-Host ""
if ($found_url) {
    Set-Clipboard -Value $found_url
    Write-Host "[BERHASIL] URL Gacha ZZZ aktif ditemukan dan berhasil disalin ke Clipboard!" -ForegroundColor Green
    Write-Host "Link siap dipaste ke dalam Web Dashboard Data Mining." -ForegroundColor Cyan
    Write-Host "URL: $($found_url.Substring(0, [math]::Min(90, $found_url.Length)))..." -ForegroundColor Gray
} elseif ($is_expired) {
    Write-Host "[PERINGATAN] Authkey ditemukan tetapi sudah Kedaluwarsa (Time Out)." -ForegroundColor Yellow
    Write-Host "Langkah Mudah:" -ForegroundColor Cyan
    Write-Host "1. Buka game Zenless Zone Zero." -ForegroundColor White
    Write-Host "2. Buka menu Signal Search (Gacha) -> Buka tab History (Riwayat)." -ForegroundColor White
    Write-Host "3. Jalankan kembali script ini." -ForegroundColor White
} else {
    Write-Host "[!] Belum ada URL history gacha yang terekam." -ForegroundColor Red
    Write-Host "Silakan buka menu Signal Search -> History di dalam game ZZZ terlebih dahulu." -ForegroundColor Yellow
}
Write-Host "==========================================================" -ForegroundColor Cyan
