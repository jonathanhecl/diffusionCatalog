#!/usr/bin/env bash
# ==============================================================================
# DiffusionCatalog — Direct Source Runner (No persistent build artifact)
# Checks required/optional dependencies, prompts to install anything missing,
# and runs the application directly via `go run ./cmd/catalogd`.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ask_install() {
  local name="$1"
  local reason="$2"
  if [[ "${CATALOG_NON_INTERACTIVE:-}" == "true" || ! -t 0 ]]; then
    return 1
  fi
  read -r -p "[?] Missing '${name}' (${reason}). Install it now? [y/N]: " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) return 0 ;;
    *) return 1 ;;
  esac
}

install_pkg() {
  local brew_pkg="$1"
  local apt_pkg="${2:-$1}"
  local os_name
  os_name="$(uname -s)"

  if [[ "$os_name" == "Darwin" ]]; then
    if ! command -v brew >/dev/null 2>&1; then
      echo "[!] Homebrew ('brew') is not installed."
      if ask_install "Homebrew" "package manager for macOS"; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      else
        echo "[!] Cannot install ${brew_pkg} without Homebrew. Exiting."
        exit 1
      fi
    fi
    brew install "$brew_pkg"
  elif [[ "$os_name" == "Linux" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y "$apt_pkg"
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y "$apt_pkg"
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm "$apt_pkg"
    else
      echo "[!] Unsupported Linux package manager. Please install '${apt_pkg}' manually."
      exit 1
    fi
  else
    echo "[!] Automatic installation not supported on ${os_name}. Please install '${brew_pkg}' manually."
    exit 1
  fi
}

# 1. Verify Go toolchain is available
if ! command -v go >/dev/null 2>&1; then
  if ask_install "go" "required to run DiffusionCatalog from source"; then
    install_pkg "go" "golang-go"
  else
    echo "[!] Go is required to run without compiling. Run ./build.sh or install Go first."
    exit 1
  fi
fi

echo "[*] Starting DiffusionCatalog from source (go run ./cmd/catalogd)..."
exec go run ./cmd/catalogd "$@"
