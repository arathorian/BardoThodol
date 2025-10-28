#!/bin/bash

# Bardo Quantum Model - GitHub Deployment Script
# Complete deployment with security and validation

set -e  # Exit on any error

echo "🚀 Starting Bardo Quantum Model GitHub Deployment"
echo "=================================================="

# Configuration
REPO_URL="https://github.com/arathorian/QuantumBardoTodol.git"
AUTHOR_NAME="arathorian"
AUTHOR_EMAIL="arathorian@users.noreply.github.com"  # Protected email
PROJECT_NAME="Bardo Quantum Model"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        log_error "Python is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check if in project directory
    if [ ! -f "README.md" ] || [ ! -d "src" ]; then
        log_error "Please run this script from the project root directory."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Security check
run_security_audit() {
    log_info "Running security audit..."
    
    # Check for exposed emails
    if grep -r "horaciohamann@gmail.com" . --exclude-dir=.git --exclude=*.pyc; then
        log_error "Personal email found in files. Please remove before deployment."
        exit 1
    fi
    
    # Check for other sensitive information
    if grep -r "password\|secret\|key" . --exclude-dir=.git --exclude=*.pyc | grep -v "#"; then
        log_warning "Potential sensitive information found. Please review."
    fi
    
    # Run Python security audit
    if [ -f "scripts/security_audit.py" ]; then
        python scripts/security_audit.py
    fi
    
    log_success "Security audit completed"
}

# Initialize Git repository
setup_git_repository() {
    log_info "Setting up Git repository..."
    
    # Initialize if not already a git repository
    if [ ! -d ".git" ]; then
        git init
        log_success "Git repository initialized"
    else
        log_info "Git repository already initialized"
    fi
    
    # Configure Git safely
    git config user.name "$AUTHOR_NAME"
    git config user.email "$AUTHOR_EMAIL"
    
    # Additional security configurations
    git config --local commit.gpgsign false  # Disable if no GPG key setup
    
    log_success "Git configuration set up securely"
}

# Run tests and validation
run_validation() {
    log_info "Running validation tests..."
    
    # Check if requirements are installed
    if [ -f "requirements.txt" ]; then
        log_info "Installing requirements..."
        pip install -r requirements.txt
    fi
    
    # Run Python tests if available
    if [ -d "tests" ]; then
        log_info "Running test suite..."
        if python -m pytest tests/ -v --tb=short; then
            log_success "All tests passed"
        else
            log_error "Tests failed. Please fix before deployment."
            exit 1
        fi
    else
        log_warning "No tests directory found"
    fi
    
    # Run code quality checks
    if command -v black &> /dev/null; then
        log_info "Formatting code with Black..."
        black src/ tests/ scripts/ --exclude=__pycache__
    fi
    
    if command -v flake8 &> /dev/null; then
        log_info "Running flake8 code style check..."
        flake8 src/ tests/ scripts/ --exclude=__pycache__ --max-line-length=100
    fi
    
    log_success "Validation completed"
}

# Create initial commit
create_initial_commit() {
    log_info "Creating initial commit..."
    
    # Add all files except those in .gitignore
    git add .
    
    # Check if there are changes to commit
    if git diff-index --quiet HEAD --; then
        log_info "No changes to commit"
        return 0
    fi
    
    # Create commit
    git commit -m "🎉 Initial commit: $PROJECT_NAME

- Quantum models with qutrits for Bardo states
- Karma dynamics simulation with scientific validation  
- ERROR 505 theoretical framework implementation
- Complete documentation and examples
- Security-protected configuration

This commit establishes the complete Bardo Quantum Model
framework for quantum computational modeling of
Tibetan Bardo Thödol states."

    log_success "Initial commit created"
}

# Setup remote repository and push
setup_remote_and_push() {
    log_info "Setting up remote repository..."
    
    # Check if remote already exists
    if git remote get-url origin &> /dev/null; then
        log_info "Remote origin already exists: $(git remote get-url origin)"
        
        # Ask if user wants to change remote
        read -p "Do you want to change the remote URL to $REPO_URL? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git remote set-url origin "$REPO_URL"
            log_success "Remote URL updated to $REPO_URL"
        fi
    else
        git remote add origin "$REPO_URL"
        log_success "Remote origin added: $REPO_URL"
    fi
    
    # Set main branch
    git branch -M main
    
    log_info "Pushing to remote repository..."
    
    # Push with retry logic
    if git push -u origin main; then
        log_success "Successfully pushed to $REPO_URL"
    else
        log_error "Failed to push to remote. Please check your credentials and network connection."
        log_info "You may need to set up GitHub credentials or use a personal access token."
        exit 1
    fi
}

# Create tags and releases
create_tags() {
    log_info "Creating version tags..."
    
    # Create initial version tag
    git tag -a v1.0.0 -m "Version 1.0.0 - Initial release

Features:
- Complete quantum model of Bardo Thödol states
- Qutrit-based state representation
- Karma dynamics simulation
- ERROR 505 analysis framework
- Scientific validation suite
- Publication-quality visualizations

This release represents the complete implementation
of the quantum computational framework for modeling
post-mortem states described in the Tibetan Book of the Dead."
    
    # Push tags
    git push --tags
    log_success "Version tags created and pushed"
}

# Final setup and instructions
final_setup() {
    log_info "Performing final setup..."
    
    # Create GitHub workflow directories if they don't exist
    mkdir -p .github/workflows
    
    # Make scripts executable
    chmod +x scripts/*.sh 2>/dev/null || true
    
    log_success "Final setup completed"
}

# Display success message
display_success() {
    echo
    echo "=================================================="
    log_success "🎉 Bardo Quantum Model Successfully Deployed!"
    echo "=================================================="
    echo
    echo "📁 Repository: $REPO_URL"
    echo "👤 Author: $AUTHOR_NAME"
    echo "🔐 Security: Email protected, validation passed"
    echo "🧪 Tests: All validation checks completed"
    echo
    echo "Next steps:"
    echo "1. Visit your repository: $REPO_URL"
    echo "2. Set up GitHub Actions for CI/CD"
    echo "3. Configure GitHub Pages for documentation"
    echo "4. Add collaborators if needed"
    echo "5. Consider setting up Zenodo for archiving"
    echo
    echo "For support or issues:"
    echo "📝 Create an issue: $REPO_URL/issues"
    echo
}

# Main deployment function
main() {
    log_info "Starting deployment of $PROJECT_NAME"
    
    # Execute deployment steps
    check_prerequisites
    run_security_audit
    setup_git_repository
    run_validation
    create_initial_commit
    setup_remote_and_push
    create_tags
    final_setup
    display_success
    
    log_success "Deployment process completed successfully!"
}

# Run main function
main "$@"