# Demo-RoomPipeline.ps1
# Minimal wrapper to run single_room_pipeline.py and produce structured demo output.
# Usage: .\Demo-RoomPipeline.ps1 -Dataset <relative_dataset_path> -Frame <frame_number> [-AssignmentRoot <path>]
# Example: .\Demo-RoomPipeline.ps1 -Dataset single_scan_floor_only/1a8384c3f6 -Frame 0
# NOTE: Run this script from the repository root. If AssignmentRoot is omitted, it defaults to a
# sibling directory relative to the repo root (assuming the standard layout).

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Dataset,

    [Parameter(Mandatory=$true)]
    [int]$Frame,

    [Parameter(Mandatory=$false)]
    [string]$AssignmentRoot
)

# Determine script and repository location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = $scriptDir  # assume script lives in repo root

# Determine assignment data root
if ($AssignmentRoot) {
    $assignmentBase = $AssignmentRoot
} else {
    # Default: try to infer as sibling of repo root
    $parent = Split-Path $repoRoot -Parent
    $assignmentBase = Join-Path $parent "Assignment - YC Startup-20261001T152812Z-1-001\Assignment - YC Startup"
    if (-not (Test-Path $assignmentBase -PathType Container)) {
        Write-Warning "Default assignment root '$assignmentBase' not found. Please provide -AssignmentRoot explicitly."
    }
}

# Construct full dataset path
$datasetPath = Join-Path $assignmentBase $Dataset
$depthDir = Join-Path $datasetPath "depth"
$intrinsicsPath = Join-Path $datasetPath "camera_matrix.csv"

# Validate inputs
if (-Not (Test-Path $depthDir -PathType Container)) {
    Write-Error "Depth directory not found: $depthDir"
    exit 1
}
if (-Not (Test-Path $intrinsicsPath -PathType Leaf)) {
    Write-Error "Camera matrix not found: $intrinsicsPath"
    exit 1
}
$depthFiles = Get-ChildItem -Path $depthDir -Filter *.png | Sort-Object Name
if ($depthFiles.Count -eq 0) {
    Write-Error "No depth images found in $depthDir"
    exit 1
}
if ($Frame -lt 0 -or $Frame -ge $depthFiles.Count) {
    Write-Error "Frame index $Frame out of range (0-$($depthFiles.Count-1))"
    exit 1
}
$selectedDepthFile = $depthFiles[$Frame]
Write-Host "Selected frame: $($selectedDepthFile.Name) (index $Frame of $($depthFiles.Count))"

# Build output directory name: output/demo_<dataset>_<frame>
# Replace path separators with underscores for safety
$datasetSlug = $Dataset -replace '[\\/]', '_'
$outputDir = Join-Path "output" ("demo_${datasetSlug}_Frame${Frame}")
# Ensure output directory does not exist or contain any entries (including hidden)
if (Test-Path $outputDir -PathType Container) {
    $items = Get-ChildItem -Path $outputDir -Force -ErrorAction SilentlyContinue
    if ($items) {
        Write-Error "Output directory already exists and is not empty: $outputDir"
        Write-Host "       Remove or rename it to avoid overwriting previous results."
        exit 1
    }
} else {
    # Create directory
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Run single_room_pipeline.py
$pyExe = "python"
$scriptPath = Join-Path $repoRoot "src/single_room_pipeline.py"
$arguments = @(
    "--dataset", $Dataset,
    "--frame", $Frame.ToString(),
    "--output-dir", $outputDir
)
Write-Host "Running: $pyExe $scriptPath $arguments"
& $pyExe $scriptPath @arguments
if ($LASTEXITCODE -ne 0) {
    Write-Error "single_room_pipeline.py failed with exit code $LASTEXITCODE"
    exit 1
}
Write-Host "Pipeline completed successfully."

# Generate run_metadata.json
$metadata = @{
    dataset_identifier = $Dataset
    selected_frame = $Frame
    input_dimensions = $null  # will fill below
    depth_scale_verified = $false
    camera_calibration_verified = $false
    geometric_measurements_units = "raw depth units (uint16)"
    known_limitations = @(
        "Depth values are raw uint16 values; physical scale is unknown."
        "Same-FOV intrinsics scaling is unverified."
        "Plane labels are unknown."
        "The distance threshold is heuristic (1% of depth range)."
        "RANSAC may fail to identify meaningful planes."
        "This is single-frame geometry, not a complete room reconstruction."
        "No measurement accuracy has been established against ground truth."
    )
}
# Get image dimensions from the selected depth file
try {
    Add-Type -AssemblyName System.Drawing
    $bmp = [System.Drawing.Bitmap]::FromFile($selectedDepthFile.FullName)
    $metadata.input_dimensions = @($bmp.Height, $bmp.Width)
    $bmp.Dispose()
} catch {
    # Fallback: if .NET fails, try to note that we couldn't get dimensions
    $metadata.known_limitations += "Could not determine input dimensions via .NET; using placeholder."
}

# Write metadata
$metadataPath = Join-Path $outputDir "run_metadata.json"
$metadataJson = $metadata | ConvertTo-Json -Depth 5
Set-Content -Path $metadataPath -Value $metadataJson -Encoding UTF8
Write-Host "Saved run metadata to: $metadataPath"

# Final summary
Write-Host "`nDemo output directory: $outputDir"
Write-Host "Contents:"
Get-ChildItem -Path $outputDir -File | ForEach-Object { Write-Host "  $($_.Name)" }