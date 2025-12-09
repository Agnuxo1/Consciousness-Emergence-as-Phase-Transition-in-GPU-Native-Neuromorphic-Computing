Param(
    [string]$OutDir = "release"
)

Write-Host "Preparing release assets..."
if (-Not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Compress-Archive -Path .\Imagen\* -DestinationPath "$OutDir\slides.zip" -Force
Compress-Archive -Path .\1-2\* -DestinationPath "$OutDir\dataset_1-2.zip" -Force
Compress-Archive -Path .\3-4\* -DestinationPath "$OutDir\dataset_3-4.zip" -Force
Compress-Archive -Path .\5-6\* -DestinationPath "$OutDir\dataset_5-6.zip" -Force

Write-Host "Copied manuscript files into release directory."
Copy-Item -Path .\"NeuroCHIMERA_ Consciousness Emergence as Phase Transition in Neuromorphic GPU-Native Computing1.pdf" -Destination $OutDir -Force
Copy-Item -Path .\NeuroCHIMERA_Paper.html -Destination $OutDir -Force

Write-Host "Release packaging complete. Files are in $OutDir"
