.PHONY: all build run test vet clean

BINARY_NAME=catalogd
BUILD_DIR=build

all: build

build:
	CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o $(BINARY_NAME) ./cmd/catalogd

run:
	go run ./cmd/catalogd

test:
	go test -v -count=1 ./...

vet:
	go vet ./...

clean:
	rm -f $(BINARY_NAME)
	rm -rf $(BUILD_DIR)
