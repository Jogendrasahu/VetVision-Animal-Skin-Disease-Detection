$edgePath = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$destDir = "C:\Users\Jogen\.gemini\antigravity\brain\bd2d1c8f-ea4a-4af7-ab2c-d59c03a9ea5d\screenshots"
if (!(Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force }

$pages = @(
    @{ Name = "vetvision_home_desktop"; Url = "http://127.0.0.1:5000/"; Width = 1280; Height = 900 },
    @{ Name = "vetvision_home_dark"; Url = "http://127.0.0.1:5000/?theme=dark"; Width = 1280; Height = 900 },
    @{ Name = "vetvision_home_tablet"; Url = "http://127.0.0.1:5000/"; Width = 768; Height = 1024 },
    @{ Name = "vetvision_home_mobile"; Url = "http://127.0.0.1:5000/"; Width = 375; Height = 812 },

    @{ Name = "vetvision_detect_desktop"; Url = "http://127.0.0.1:5000/detect"; Width = 1280; Height = 950 },
    @{ Name = "vetvision_detect_dark"; Url = "http://127.0.0.1:5000/detect?theme=dark"; Width = 1280; Height = 950 },
    @{ Name = "vetvision_detect_tablet"; Url = "http://127.0.0.1:5000/detect"; Width = 768; Height = 1024 },
    @{ Name = "vetvision_detect_mobile"; Url = "http://127.0.0.1:5000/detect"; Width = 375; Height = 812 },

    @{ Name = "vetvision_diseases_desktop"; Url = "http://127.0.0.1:5000/diseases"; Width = 1280; Height = 950 },
    @{ Name = "vetvision_dataset_desktop"; Url = "http://127.0.0.1:5000/dataset"; Width = 1280; Height = 950 },
    @{ Name = "vetvision_history_desktop"; Url = "http://127.0.0.1:5000/history"; Width = 1280; Height = 950 },
    @{ Name = "vetvision_guide_desktop"; Url = "http://127.0.0.1:5000/guide"; Width = 1280; Height = 950 },
    @{ Name = "vetvision_about_desktop"; Url = "http://127.0.0.1:5000/about"; Width = 1280; Height = 950 },

    @{ Name = "vetvision_result_dashboard_light"; Url = "http://127.0.0.1:5000/detect-preview"; Width = 1280; Height = 1000 },
    @{ Name = "vetvision_result_dashboard_dark"; Url = "http://127.0.0.1:5000/detect-preview?theme=dark"; Width = 1280; Height = 1000 },
    @{ Name = "vetvision_result_dashboard_mobile"; Url = "http://127.0.0.1:5000/detect-preview"; Width = 375; Height = 812 }
)

foreach ($p in $pages) {
    $tempFile = "$env:TEMP\$($p.Name).png"
    $targetFile = "$destDir\$($p.Name).png"
    $userDir = "$env:TEMP\edge_shot_$($p.Name)"
    
    $argsList = @(
        "--headless",
        "--disable-gpu",
        "--window-size=$($p.Width),$($p.Height)",
        "--user-data-dir=$userDir",
        "--screenshot=$tempFile",
        $p.Url
    )
    
    Start-Process -FilePath $edgePath -ArgumentList $argsList -Wait
    if (Test-Path $tempFile) {
        Copy-Item $tempFile -Destination $targetFile -Force
        Remove-Item $tempFile -Force
        Write-Host "Captured: $($p.Name).png"
    } else {
        Write-Host "Failed to capture: $($p.Name)"
    }
}

Get-ChildItem -Path $destDir -Filter "vetvision_*.png" | Select-Object Name, Length, LastWriteTime

