package metadata

import (
	"bytes"
	"encoding/binary"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

var typeSizes = map[uint16]uint32{
	1:  1, // BYTE
	2:  1, // ASCII
	3:  2, // SHORT
	4:  4, // LONG
	5:  8, // RATIONAL
	6:  1, // SBYTE
	7:  1, // UNDEFINED
	8:  2, // SSHORT
	9:  4, // SLONG
	10: 8, // SRATIONAL
	11: 4, // FLOAT
	12: 8, // DOUBLE
}

type tagEntry struct {
	tagType uint16
	count   uint32
	raw4    []byte
}

// parseEXIFTags parses raw TIFF bytes and returns a map of tagID -> tagEntry and byte order.
func parseEXIFTags(tiff []byte) (map[uint16]tagEntry, binary.ByteOrder) {
	// Strip "Exif\x00\x00" header if present
	if len(tiff) >= 6 && bytes.Equal(tiff[:6], []byte("Exif\x00\x00")) {
		tiff = tiff[6:]
	}

	tags := make(map[uint16]tagEntry)
	if len(tiff) < 8 {
		return tags, nil
	}

	var bo binary.ByteOrder
	if tiff[0] == 'I' && tiff[1] == 'I' {
		bo = binary.LittleEndian
	} else if tiff[0] == 'M' && tiff[1] == 'M' {
		bo = binary.BigEndian
	} else {
		return tags, nil
	}

	magic := bo.Uint16(tiff[2:4])
	if magic != 42 {
		return tags, bo
	}

	ifd0Offset := int(bo.Uint32(tiff[4:8]))
	queue := []int{ifd0Offset}
	seen := make(map[int]bool)

	for len(queue) > 0 {
		off := queue[len(queue)-1]
		queue = queue[:len(queue)-1]

		if off <= 0 || seen[off] || off+2 > len(tiff) {
			continue
		}
		seen[off] = true

		numEntries := int(bo.Uint16(tiff[off : off+2]))
		for i := 0; i < numEntries; i++ {
			entryOff := off + 2 + i*12
			if entryOff+12 > len(tiff) {
				break
			}
			tag := bo.Uint16(tiff[entryOff : entryOff+2])
			typ := bo.Uint16(tiff[entryOff+2 : entryOff+4])
			cnt := bo.Uint32(tiff[entryOff+4 : entryOff+8])
			raw4 := append([]byte(nil), tiff[entryOff+8:entryOff+12]...)

			tags[tag] = tagEntry{
				tagType: typ,
				count:   cnt,
				raw4:    raw4,
			}

			// Sub-IFD pointers: Exif IFD (0x8769) and Interoperability IFD (0xA005)
			if tag == 0x8769 || tag == 0xA005 {
				subOff := int(bo.Uint32(raw4))
				queue = append(queue, subOff)
			}
		}

		nextOffsetPos := off + 2 + numEntries*12
		if nextOffsetPos+4 <= len(tiff) {
			nextIFD := int(bo.Uint32(tiff[nextOffsetPos : nextOffsetPos+4]))
			if nextIFD > 0 {
				queue = append(queue, nextIFD)
			}
		}
	}

	return tags, bo
}

func getTagBytes(tiff []byte, entry tagEntry, bo binary.ByteOrder) []byte {
	szPerItem, ok := typeSizes[entry.tagType]
	if !ok {
		szPerItem = 1
	}
	totalSize := int(szPerItem * entry.count)
	if totalSize <= 4 {
		if totalSize > len(entry.raw4) {
			totalSize = len(entry.raw4)
		}
		return entry.raw4[:totalSize]
	}

	off := int(bo.Uint32(entry.raw4))
	if off < 0 || off >= len(tiff) {
		return nil
	}
	end := off + totalSize
	if end > len(tiff) {
		end = len(tiff)
	}
	return tiff[off:end]
}

// decodeUserComment decodes EXIF UserComment tag (0x9286) payload.
func decodeUserComment(raw []byte) string {
	if len(raw) < 8 {
		return cleanNullAndSpace(string(raw))
	}

	head := strings.ToUpper(string(raw[:8]))
	body := raw[8:]

	if strings.HasPrefix(head, "UNICODE") {
		// Check BOM
		if len(body) >= 2 {
			if body[0] == 0xFF && body[1] == 0xFE {
				return cleanNullAndSpace(decodeUTF16(body[2:], binary.LittleEndian))
			} else if body[0] == 0xFE && body[1] == 0xFF {
				return cleanNullAndSpace(decodeUTF16(body[2:], binary.BigEndian))
			}
		}

		// Try both UTF-16LE and UTF-16BE, pick highest printable score
		scoreLE, sLE := utf16Score(body, binary.LittleEndian)
		scoreBE, sBE := utf16Score(body, binary.BigEndian)

		if scoreLE >= scoreBE && scoreLE > 0 {
			return cleanNullAndSpace(sLE)
		} else if scoreBE > scoreLE && scoreBE > 0 {
			return cleanNullAndSpace(sBE)
		}

		// Fallback to UTF-16LE
		return cleanNullAndSpace(decodeUTF16(body, binary.LittleEndian))
	}

	// ASCII or undefined raw bytes
	return cleanNullAndSpace(string(body))
}

func decodeUTF16(b []byte, bo binary.ByteOrder) string {
	if len(b)%2 != 0 {
		b = b[:len(b)-1]
	}
	u16s := make([]uint16, len(b)/2)
	for i := 0; i < len(u16s); i++ {
		u16s[i] = bo.Uint16(b[i*2 : i*2+2])
	}
	runes := utf16.Decode(u16s)
	return strings.TrimPrefix(string(runes), "\uFEFF")
}

func utf16Score(b []byte, bo binary.ByteOrder) (int, string) {
	s := decodeUTF16(b, bo)
	score := 0
	for _, r := range s {
		if r != utf8.RuneError && ((r >= 32 && r <= 126) || r == '\n' || r == '\r' || r == '\t') {
			score++
		}
	}
	return score, s
}

func cleanNullAndSpace(s string) string {
	s = strings.Trim(s, "\x00")
	return strings.TrimSpace(s)
}

// extractEXIFStrings extracts all meaningful candidate strings from EXIF payloads.
func extractEXIFStrings(payloads [][]byte) []string {
	var texts []string
	for _, p := range payloads {
		if len(p) >= 6 && bytes.Equal(p[:6], []byte("Exif\x00\x00")) {
			p = p[6:]
		}
		tags, bo := parseEXIFTags(p)
		if bo == nil {
			continue
		}

		// Tags to check:
		// 0x9286 = UserComment
		// 0x010E = ImageDescription
		// 0x010F = Make
		// 0x0110 = Model
		// 0x0131 = Software
		// 0x9C9B = XPTitle
		// 0x9C9C = XPComment
		tagOrder := []uint16{0x9286, 0x010E, 0x010F, 0x0110, 0x0131, 0x9C9B, 0x9C9C}
		for _, tagID := range tagOrder {
			entry, ok := tags[tagID]
			if !ok {
				continue
			}
			raw := getTagBytes(p, entry, bo)
			if len(raw) == 0 {
				continue
			}

			var s string
			switch tagID {
			case 0x9286:
				s = decodeUserComment(raw)
			case 0x9C9B, 0x9C9C:
				s = decodeUTF16(raw, binary.LittleEndian)
				s = cleanNullAndSpace(s)
			default:
				s = cleanNullAndSpace(string(raw))
			}

			if s != "" {
				texts = append(texts, s)
			}
		}
	}
	return texts
}
