# PowerShell deployment script for Windows

# Enable verbose output
$VerbosePreference = "Continue"

Write-Host "Starting deployment process..." -ForegroundColor Green

# Check if fly.io CLI is installed
if (-not (Get-Command fly -ErrorAction SilentlyContinue)) {
    Write-Host "Installing fly.io CLI..." -ForegroundColor Yellow
    # Download and install fly.io CLI
    $flyUrl = "https://fly.io/docs/hands-on/install-flyctl/"
    Write-Host "Please install fly.io CLI from: $flyUrl"
    Write-Host "After installation, please restart this script."
    exit
}

# Check if logged in to fly.io
Write-Host "Checking fly.io login status..." -ForegroundColor Cyan
$loginStatus = fly auth whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Please login to fly.io first:" -ForegroundColor Yellow
    fly auth login
}

# Create app if it doesn't exist
Write-Host "Checking if app exists..." -ForegroundColor Cyan
$appExists = fly apps list | Select-String "medical-knowledge-base"
if (-not $appExists) {
    Write-Host "Creating new fly.io app..." -ForegroundColor Yellow
    fly apps create medical-knowledge-base
}

# Set OpenAI API key
Write-Host "Setting OpenAI API key..." -ForegroundColor Cyan
$apiKey = Read-Host "Enter your OpenAI API key"
fly secrets set OPENAI_API_KEY=$apiKey

# Create index_data directory if it doesn't exist
if (-not (Test-Path "index_data")) {
    Write-Host "Creating index_data directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "index_data"
}

# Check if index data exists
if (-not (Test-Path "index_data\vector_store")) {
    Write-Host "Warning: No index data found. Please run local_indexer.py first." -ForegroundColor Red
    exit
}

# Check if Docker is installed
Write-Host "Checking Docker installation..." -ForegroundColor Cyan
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Please install Docker Desktop for Windows first." -ForegroundColor Red
    Write-Host "Download from: https://www.docker.com/products/docker-desktop"
    exit
}

# Check if Docker is running
Write-Host "Checking if Docker is running..." -ForegroundColor Cyan
$dockerStatus = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit
}

# Deploy the application
Write-Host "Deploying to fly.io..." -ForegroundColor Green
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

# Run fly deploy with verbose output
$deployOutput = fly deploy --verbose 2>&1
$deployStatus = $LASTEXITCODE

if ($deployStatus -eq 0) {
    Write-Host "Deployment completed successfully!" -ForegroundColor Green
    Write-Host "Your app should be available at: https://medical-knowledge-base.fly.dev" -ForegroundColor Green
    
    # Check app status
    Write-Host "Checking app status..." -ForegroundColor Cyan
    fly status
} else {
    Write-Host "Deployment failed with error:" -ForegroundColor Red
    Write-Host $deployOutput -ForegroundColor Red
    Write-Host "Please check the error message above and try again." -ForegroundColor Red
} 