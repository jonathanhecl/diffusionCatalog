package metadata

import (
	"encoding/binary"
	"os"
)

// ExtractJPEGEXIFPayloads scans JPEG markers to extract APP1 EXIF segment payloads.
func ExtractJPEGEXIFPayloads(data []byte) [][]byte {
	var payloads [][]byte
	if len(data) < 4 || data[0] != 0xFF || data[1] != 0xD8 {
		return payloads
	}

	pos := 2
	for pos+4 <= len(data) {
		if data[pos] != 0xFF {
			break
		}
		// Skip padding FF bytes if any
		for pos < len(data) && data[pos] == 0xFF {
			pos++
		}
		if pos >= len(data) {
			break
		}
		marker := data[pos]
		pos++

		if marker == 0xD8 { // SOI
			continue
		}
		if marker == 0xDA || marker == 0xD9 { // SOS (Start of Scan) or EOI
			break
		}
		if pos+2 > len(data) {
			break
		}
		segLen := int(binary.BigEndian.Uint16(data[pos : pos+2]))
		if segLen < 2 || pos+segLen > len(data) {
			break
		}

		segData := data[pos+2 : pos+segLen]
		if marker == 0xE1 { // APP1
			if len(segData) >= 6 && string(segData[:6]) == "Exif\x00\x00" {
				payloads = append(payloads, segData[6:])
			} else {
				payloads = append(payloads, segData)
			}
		}

		pos += segLen
	}
	return payloads
}

// ExtractJPEGEXIFFromFile reads a JPEG file and extracts EXIF payloads.
func ExtractJPEGEXIFFromFile(path string) ([][]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ExtractJPEGEXIFPayloads(data), nil
}
