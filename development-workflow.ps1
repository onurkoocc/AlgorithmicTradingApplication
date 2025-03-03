# PowerShell Development Workflow Script for Algorithmic Trading Application

# Colors for better output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Blue "Algorithmic Trading Development Workflow"
Write-Output "====================================="

# Function to build the base image (only needed once or when dependencies change)
function Build-BaseImage {
    Write-ColorOutput Yellow "Building base image with dependencies..."
    Write-Output "This might take 15-20 minutes the first time, but only needs to be done once."

    # Build the base image with all dependencies
    docker build --target base -t algorithmic-trading-base:latest -f Dockerfile.optimized .

    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "Base image built successfully!"
    }
    else {
        Write-ColorOutput Red "Failed to build base image"
        exit 1
    }
}

# Function to build the final application image
function Build-AppImage {
    Write-ColorOutput Yellow "Building application image..."

    # Build the final application image using the pre-built base
    docker build --cache-from algorithmic-trading-base:latest -t algorithmic-trading-app:latest -f Dockerfile.optimized .

    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "Application image built successfully!"
    }
    else {
        Write-ColorOutput Red "Failed to build application image"
        exit 1
    }
}

# Function to start development environment
function Start-DevEnvironment {
    Write-ColorOutput Yellow "Starting development environment..."

    # Check if the container is already running
    $runningContainer = docker ps -q -f name=algorithmic-trading-container
    if ($runningContainer) {
        Write-ColorOutput Blue "Development container already running"
    }
    else {
        # Start the container with development compose file
        docker-compose -f docker-compose.dev.yml up -d

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput Green "Development environment started!"
        }
        else {
            Write-ColorOutput Red "Failed to start development environment"
            exit 1
        }
    }
}

# Function to run the application in development mode
function Run-App {
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        $arguments
    )

    Write-ColorOutput Yellow "Running application in development mode..."

    # Construct the command string with arguments
    $cmdArgs = $arguments -join ' '

    # Execute the application in the running container
    docker exec -it algorithmic-trading-container python3 /app/main.py $cmdArgs
}

# Function to show logs
function Show-Logs {
    Write-ColorOutput Yellow "Showing container logs..."
    docker logs -f algorithmic-trading-container
}

# Function to enter the container shell
function Enter-Shell {
    Write-ColorOutput Yellow "Opening shell in the container..."
    docker exec -it algorithmic-trading-container /bin/bash
}

# Function to stop the development environment
function Stop-DevEnvironment {
    Write-ColorOutput Yellow "Stopping development environment..."
    docker-compose -f docker-compose.dev.yml down

    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "Development environment stopped!"
    }
    else {
        Write-ColorOutput Red "Failed to stop development environment"
        exit 1
    }
}

# Show help
function Show-Help {
    Write-ColorOutput Blue "Development Workflow Commands:"
    Write-Output "  base-build    - Build base image with dependencies (only needed once or when dependencies change)"
    Write-Output "  app-build     - Build application image"
    Write-Output "  start         - Start development environment"
    Write-Output "  run [args]    - Run application in development mode (with optional arguments)"
    Write-Output "  logs          - Show container logs"
    Write-Output "  shell         - Open shell in the container"
    Write-Output "  stop          - Stop development environment"
    Write-Output "  help          - Show this help message"
}

# Parse command line arguments
$command = $args[0]
switch ($command) {
    "base-build" {
        Build-BaseImage
    }
    "app-build" {
        Build-AppImage
    }
    "start" {
        Start-DevEnvironment
    }
    "run" {
        $runArgs = $args[1..$args.Length]
        Run-App $runArgs
    }
    "logs" {
        Show-Logs
    }
    "shell" {
        Enter-Shell
    }
    "stop" {
        Stop-DevEnvironment
    }
    default {
        Show-Help
    }
}