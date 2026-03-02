# PowerShell script to run PR commit extraction with auto-retry
$maxAttempts = 100
$attempt = 1

while ($attempt -le $maxAttempts) {
    Write-Host "`n=== Attempt $attempt/$maxAttempts ===" -ForegroundColor Cyan
    
    python src/extract_pr_commits.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nExtraction completed successfully!" -ForegroundColor Green
        
        # Verify the output
        $result = python -c "import pandas as pd; df=pd.read_csv(r'Repo data/flink_pr_commits.csv'); print(f'{len(df)}/5125')"
        Write-Host "Final PR count: $result" -ForegroundColor Green
        break
    }
    
    Write-Host "`nAttempt $attempt failed. Retrying in 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    $attempt++
}

if ($attempt -gt $maxAttempts) {
    Write-Host "`nFailed after $maxAttempts attempts." -ForegroundColor Red
    exit 1
}
