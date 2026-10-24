package metadata

import (
	"encoding/binary"
	"errors"
	"io"
	"os"
	"strings"
)

// ParseMP4File opens an MP4 or MOV file and extracts diffusion metadata tags (parameters, prompt, workflow)
// and video dimensions (width, height) without loading large video data (mdat) into memory.
func ParseMP4File(filePath string) (map[string]string, int, int, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	return ParseMP4(f)
}

// ParseMP4 parses an MP4/QuickTime ISO-BMFF stream from an io.ReadSeeker.
func ParseMP4(rs io.ReadSeeker) (map[string]string, int, int, error) {
	tags := make(map[string]string)
	var width, height int

	for {
		boxType, boxDataSize, err := readBoxHeader(rs)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return tags, width, height, err
		}

		contentStart, _ := rs.Seek(0, io.SeekCurrent)
		boxEnd := contentStart + int64(boxDataSize)

		if boxType == "moov" {
			w, h := parseMoovBox(rs, boxEnd, tags)
			if w > 0 && h > 0 && width == 0 {
				width = w
				height = h
			}
		}

		if _, err := rs.Seek(boxEnd, io.SeekStart); err != nil {
			break
		}
	}

	return tags, width, height, nil
}

func parseMoovBox(rs io.ReadSeeker, endPos int64, tags map[string]string) (int, int) {
	var width, height int
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		boxType, dataSize, err := readBoxHeader(rs)
		if err != nil {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)
		subEnd := contentPos + int64(dataSize)

		switch boxType {
		case "trak":
			w, h := parseTrakBox(rs, subEnd)
			if w > 0 && h > 0 && width == 0 {
				width = w
				height = h
			}
		case "udta":
			parseUdtaBox(rs, subEnd, tags)
		case "meta":
			parseMetaBox(rs, subEnd, tags)
		}

		rs.Seek(subEnd, io.SeekStart)
	}
	return width, height
}

func parseTrakBox(rs io.ReadSeeker, endPos int64) (int, int) {
	var width, height int
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		boxType, dataSize, err := readBoxHeader(rs)
		if err != nil {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)
		subEnd := contentPos + int64(dataSize)

		if boxType == "tkhd" {
			w, h := parseTkhdBox(rs, subEnd)
			if w > 0 && h > 0 {
				width = w
				height = h
			}
		}

		rs.Seek(subEnd, io.SeekStart)
	}
	return width, height
}

func parseTkhdBox(rs io.ReadSeeker, endPos int64) (int, int) {
	// Width and Height are the last 8 bytes of the tkhd box, as 16.16 fixed-point big-endian ints
	if _, err := rs.Seek(endPos-8, io.SeekStart); err != nil {
		return 0, 0
	}
	var buf [8]byte
	if _, err := io.ReadFull(rs, buf[:]); err != nil {
		return 0, 0
	}
	wFixed := binary.BigEndian.Uint32(buf[0:4])
	hFixed := binary.BigEndian.Uint32(buf[4:8])
	w := int(wFixed >> 16)
	h := int(hFixed >> 16)
	return w, h
}

func parseUdtaBox(rs io.ReadSeeker, endPos int64, tags map[string]string) {
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		boxType, dataSize, err := readBoxHeader(rs)
		if err != nil {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)
		subEnd := contentPos + int64(dataSize)

		if boxType == "meta" {
			parseMetaBox(rs, subEnd, tags)
		}

		rs.Seek(subEnd, io.SeekStart)
	}
}

func parseMetaBox(rs io.ReadSeeker, endPos int64, tags map[string]string) {
	// MP4 meta box can be a fullbox (4 bytes version+flags) or raw box.
	// Inspect next bytes to determine if version/flags are present.
	var probe [8]byte
	if _, err := io.ReadFull(rs, probe[:]); err != nil {
		return
	}

	// Check if probe[4:8] is a known box name like "hdlr", "keys", "ilst"
	firstFourCC := string(probe[4:8])
	if firstFourCC == "hdlr" || firstFourCC == "keys" || firstFourCC == "ilst" {
		// No version/flags, probe was the first sub-box header
		rs.Seek(-8, io.SeekCurrent)
	} else {
		// First 4 bytes were version+flags. Check if bytes 4:8 were part of next box
		rs.Seek(-4, io.SeekCurrent)
	}

	var keyNames []string

	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		boxType, dataSize, err := readBoxHeader(rs)
		if err != nil {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)
		subEnd := contentPos + int64(dataSize)

		switch boxType {
		case "keys":
			keyNames = parseKeysBox(rs, subEnd)
		case "ilst":
			parseIlstBox(rs, subEnd, keyNames, tags)
		}

		rs.Seek(subEnd, io.SeekStart)
	}
}

func parseKeysBox(rs io.ReadSeeker, endPos int64) []string {
	var keyNames []string
	var hdr [8]byte // 4 bytes version+flags, 4 bytes entry_count
	if _, err := io.ReadFull(rs, hdr[:]); err != nil {
		return nil
	}
	entryCount := int(binary.BigEndian.Uint32(hdr[4:8]))
	for i := 0; i < entryCount; i++ {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		var entHdr [8]byte // 4 bytes size, 4 bytes namespace (e.g. "mdta")
		if _, err := io.ReadFull(rs, entHdr[:]); err != nil {
			break
		}
		entSize := int(binary.BigEndian.Uint32(entHdr[0:4]))
		if entSize < 8 {
			break
		}
		nameLen := entSize - 8
		if nameLen > 4096 {
			nameLen = 4096
		}
		nameBuf := make([]byte, nameLen)
		if _, err := io.ReadFull(rs, nameBuf); err != nil {
			break
		}
		keyNames = append(keyNames, string(nameBuf))
		rs.Seek(curr+int64(entSize), io.SeekStart)
	}
	return keyNames
}

func parseIlstBox(rs io.ReadSeeker, endPos int64, keyNames []string, tags map[string]string) {
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		var itemHdr [8]byte
		if _, err := io.ReadFull(rs, itemHdr[:]); err != nil {
			break
		}
		itemSize := int(binary.BigEndian.Uint32(itemHdr[0:4]))
		if itemSize < 8 {
			break
		}
		itemEnd := curr + int64(itemSize)

		itemIndex := binary.BigEndian.Uint32(itemHdr[4:8])
		itemType := string(itemHdr[4:8])

		var keyName string
		if itemIndex >= 1 && int(itemIndex) <= len(keyNames) {
			keyName = keyNames[itemIndex-1]
		} else {
			keyName = strings.Trim(itemType, "\x00©")
		}

		// Find "data" sub-box inside item
		for {
			dCurr, err := rs.Seek(0, io.SeekCurrent)
			if err != nil || dCurr >= itemEnd {
				break
			}
			dType, dSize, err := readBoxHeader(rs)
			if err != nil {
				break
			}
			dContentPos, _ := rs.Seek(0, io.SeekCurrent)
			dEnd := dContentPos + int64(dSize)

			if dType == "data" && dSize >= 8 {
				// data header: 4 bytes type indicator, 4 bytes locale
				var dataMeta [8]byte
				if _, err := io.ReadFull(rs, dataMeta[:]); err == nil {
					payloadLen := int(dSize) - 8
					if payloadLen > 0 && payloadLen <= 64*1024*1024 {
						buf := make([]byte, payloadLen)
						if _, err := io.ReadFull(rs, buf); err == nil {
							tags[keyName] = string(buf)
						}
					}
				}
			}

			rs.Seek(dEnd, io.SeekStart)
		}

		rs.Seek(itemEnd, io.SeekStart)
	}
}

func readBoxHeader(rs io.ReadSeeker) (string, uint64, error) {
	var hdr [8]byte
	if _, err := io.ReadFull(rs, hdr[:]); err != nil {
		return "", 0, err
	}
	size32 := binary.BigEndian.Uint32(hdr[0:4])
	boxType := string(hdr[4:8])

	if size32 == 1 {
		// 64-bit size
		var extSize [8]byte
		if _, err := io.ReadFull(rs, extSize[:]); err != nil {
			return "", 0, err
		}
		size64 := binary.BigEndian.Uint64(extSize[:])
		if size64 < 16 {
			return boxType, 0, errors.New("invalid extended box size")
		}
		return boxType, size64 - 16, nil
	} else if size32 == 0 {
		// Box extends to end of file
		curr, _ := rs.Seek(0, io.SeekCurrent)
		end, _ := rs.Seek(0, io.SeekEnd)
		rs.Seek(curr, io.SeekStart)
		return boxType, uint64(end - curr), nil
	}

	if size32 < 8 {
		return boxType, 0, errors.New("invalid box size")
	}
	return boxType, uint64(size32 - 8), nil
}
