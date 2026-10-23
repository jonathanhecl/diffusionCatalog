#!/usr/bin/env bash
# ==============================================================================
# DiffusionCatalog — Portable & Self-Installing Binary Builder
# Compiles a 100% static Go binary (CGO_ENABLED=0) with embedded UI templates,
# CSS, and the embedded Python Vision Worker bootstrapper.
# Outputs:
#   - ./catalogd                (root workspace binary)
#   - ./build/catalogd          (portable distribution binary)
#   - ./build/start-portable.sh (portable self-checking launcher)
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

# 1. Ensure Go compiler is installed
if ! command -v go >/dev/null 2>&1; then
  if ask_install "go" "required to compile the portable DiffusionCatalog binary"; then
    install_pkg "go" "golang-go"
  else
    echo "[!] Go compiler is required to build DiffusionCatalog."
    exit 1
  fi
fi

BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "$BUILD_DIR"

echo "[*] Building portable single-file binary (CGO_ENABLED=0)..."
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o "${SCRIPT_DIR}/catalogd" ./cmd/catalogd
cp "${SCRIPT_DIR}/catalogd" "${BUILD_DIR}/catalogd"
chmod +x "${SCRIPT_DIR}/catalogd" "${BUILD_DIR}/catalogd"

# Create portable launcher inside ./build/
cat > "${BUILD_DIR}/start-portable.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [[ ! -x "./catalogd" ]]; then
  chmod +x "./catalogd"
fi

exec ./catalogd "$@"
EOF
chmod +x "${BUILD_DIR}/start-portable.sh"

echo "[+] Build complete!"
echo "    - Root binary:     ${SCRIPT_DIR}/catalogd"
echo "    - Portable folder: ${BUILD_DIR}/catalogd"
echo "    - Run directly:    ./catalogd (or ./catalogd --setup to re-run dependency installer)"
