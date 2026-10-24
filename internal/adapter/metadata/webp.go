package metadata

import (
	"encoding/binary"
	"os"
)

// ExtractWebPEXIFPayloads scans RIFF WebP chunks to locate and extract EXIF chunk payload.
func ExtractWebPEXIFPayloads(data []byte) [][]byte {
	var payloads [][]byte
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		return payloads
	}

	pos := 12
	for pos+8 <= len(data) {
		fourcc := string(data[pos : pos+4])
		chunkLen := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		if pos+8+chunkLen > len(data) {
			break
		}

		chunk := data[pos+8 : pos+8+chunkLen]
		if fourcc == "EXIF" {
			if len(chunk) >= 6 && string(chunk[:6]) == "Exif\x00\x00" {
				payloads = append(payloads, chunk[6:])
			} else {
				payloads = append(payloads, chunk)
			}
		}

		// RIFF chunks are padded to even byte length
		pos += 8 + chunkLen + (chunkLen & 1)
	}

	return payloads
}

// ExtractWebPEXIFFromFile reads a WebP file and extracts EXIF payloads.
func ExtractWebPEXIFFromFile(path string) ([][]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ExtractWebPEXIFPayloads(data), nil
}
