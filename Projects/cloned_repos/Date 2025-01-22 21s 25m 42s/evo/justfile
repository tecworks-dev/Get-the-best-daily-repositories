# Build and run the application with arguments
run args:
    just build
    bin/evo {{args}}

# List available commands
default:
    @just --list

# Build the evo binary
build:
    go build -o bin/evo ./cmd/evo

# Run all tests
test:
    go test ./...

# Run tests with coverage
test-coverage:
    go test -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out

# Clean build artifacts
clean:
    rm -rf bin/
    rm -f coverage.out

# Install development dependencies
setup:
    go mod download
    go mod tidy

# Run linter
lint:
    go vet ./...

# Format code
fmt:
    go fmt ./...
