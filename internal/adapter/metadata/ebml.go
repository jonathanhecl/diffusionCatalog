package metadata

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// EBML and Matroska element IDs
const (
	idEBMLHeader  = 0x1A45DFA3
	idSegment     = 0x18538067
	idSeekHead    = 0x114D9B74
	idInfo        = 0x1549A966
	idTracks      = 0x1654AE6B
	idTrackEntry  = 0xAE
	idTrackType   = 0x83
	idVideo       = 0xE0
	idPixelWidth  = 0xB0
	idPixelHeight = 0xBA
	idTags        = 0x1254C367
	idTag         = 0x7373
	idSimpleTag   = 0x67C8
	idTagName     = 0x45A3
	idTagString   = 0x4487
	idTagBinary   = 0x4485
	idCluster     = 0x1F43B675

	ebmlUnknownSize = math.MaxUint64
)

// ParseEBMLFile opens a WebM/MKV file and extracts metadata tags and video dimensions.
func ParseEBMLFile(filePath string) (map[string]string, int, int, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	return ParseEBML(f)
}

// ParseEBML parses an EBML stream (WebM/Matroska) from an io.ReadSeeker.
func ParseEBML(rs io.ReadSeeker) (map[string]string, int, int, error) {
	// 1. Validate EBML Header
	headerID, _, err := readVINT(rs, false)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read EBML header ID: %w", err)
	}
	if headerID != idEBMLHeader {
		return nil, 0, 0, errors.New("not a valid EBML file: invalid header ID")
	}

	headerSize, _, err := readVINT(rs, true)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read EBML header size: %w", err)
	}
	if _, err := rs.Seek(int64(headerSize), io.SeekCurrent); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to seek past EBML header: %w", err)
	}

	// 2. Read Segment element
	segmentID, _, err := readVINT(rs, false)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read Segment ID: %w", err)
	}
	if segmentID != idSegment {
		return nil, 0, 0, fmt.Errorf("expected Segment ID (0x18538067), got 0x%X", segmentID)
	}

	segmentSize, _, err := readVINT(rs, true)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read Segment size: %w", err)
	}

	segStart, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, 0, 0, err
	}

	var segEnd int64 = math.MaxInt64
	if segmentSize != ebmlUnknownSize {
		segEnd = segStart + int64(segmentSize)
	}

	tags := make(map[string]string)
	var width, height int

	// 3. Scan Segment children
	for {
		currPos, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || currPos >= segEnd {
			break
		}

		elemID, _, err := readVINT(rs, false)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return tags, width, height, err
		}

		elemSize, _, err := readVINT(rs, true)
		if err != nil {
			return tags, width, height, err
		}

		elemContentPos, err := rs.Seek(0, io.SeekCurrent)
		if err != nil {
			return tags, width, height, err
		}

		switch elemID {
		case idTags:
			tagEnd := elemContentPos + int64(elemSize)
			parseTagsElement(rs, tagEnd, tags)
		case idTracks:
			tracksEnd := elemContentPos + int64(elemSize)
			w, h := parseTracksElement(rs, tracksEnd)
			if w > 0 && h > 0 && width == 0 {
				width = w
				height = h
			}
		}

		// Fast forward to next top-level element
		nextPos := elemContentPos + int64(elemSize)
		if elemSize == ebmlUnknownSize {
			// If element has unknown size, we cannot easily seek past it
			break
		}
		if _, err := rs.Seek(nextPos, io.SeekStart); err != nil {
			break
		}
	}

	return tags, width, height, nil
}

func parseTagsElement(rs io.ReadSeeker, endPos int64, tags map[string]string) {
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		elemID, _, err := readVINT(rs, false)
		if err != nil {
			break
		}
		elemSize, _, err := readVINT(rs, true)
		if err != nil || elemSize == ebmlUnknownSize {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)
		if elemID == idTag {
			parseSingleTag(rs, contentPos+int64(elemSize), tags)
		}
		rs.Seek(contentPos+int64(elemSize), io.SeekStart)
	}
}

func parseSingleTag(rs io.ReadSeeker, endPos int64, tags map[string]string) {
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		elemID, _, err := readVINT(rs, false)
		if err != nil {
			break
		}
		elemSize, _, err := readVINT(rs, true)
		if err != nil || elemSize == ebmlUnknownSize {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)
		if elemID == idSimpleTag {
			name, val := parseSimpleTag(rs, contentPos+int64(elemSize))
			if name != "" && val != "" {
				tags[name] = val
			}
		}
		rs.Seek(contentPos+int64(elemSize), io.SeekStart)
	}
}

func parseSimpleTag(rs io.ReadSeeker, endPos int64) (string, string) {
	var name, strVal string
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		elemID, _, err := readVINT(rs, false)
		if err != nil {
			break
		}
		elemSize, _, err := readVINT(rs, true)
		if err != nil || elemSize == ebmlUnknownSize {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)

		switch elemID {
		case idTagName:
			if s, err := readString(rs, elemSize); err == nil {
				name = s
			}
		case idTagString:
			if s, err := readString(rs, elemSize); err == nil {
				strVal = s
			}
		case idSimpleTag:
			// Nested SimpleTag
			nestedName, nestedVal := parseSimpleTag(rs, contentPos+int64(elemSize))
			if nestedName != "" && nestedVal != "" {
				name = nestedName
				strVal = nestedVal
			}
		}
		rs.Seek(contentPos+int64(elemSize), io.SeekStart)
	}
	return name, strVal
}

func parseTracksElement(rs io.ReadSeeker, endPos int64) (int, int) {
	var width, height int
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		elemID, _, err := readVINT(rs, false)
		if err != nil {
			break
		}
		elemSize, _, err := readVINT(rs, true)
		if err != nil || elemSize == ebmlUnknownSize {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)
		if elemID == idTrackEntry {
			w, h := parseTrackEntry(rs, contentPos+int64(elemSize))
			if w > 0 && h > 0 && width == 0 {
				width = w
				height = h
			}
		}
		rs.Seek(contentPos+int64(elemSize), io.SeekStart)
	}
	return width, height
}

func parseTrackEntry(rs io.ReadSeeker, endPos int64) (int, int) {
	var isVideo bool
	var width, height int
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		elemID, _, err := readVINT(rs, false)
		if err != nil {
			break
		}
		elemSize, _, err := readVINT(rs, true)
		if err != nil || elemSize == ebmlUnknownSize {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)

		switch elemID {
		case idTrackType:
			val, err := readUint(rs, elemSize)
			if err == nil && val == 1 { // 1 = Video track
				isVideo = true
			}
		case idVideo:
			w, h := parseVideoSettings(rs, contentPos+int64(elemSize))
			if w > 0 && h > 0 {
				width = w
				height = h
			}
		}
		rs.Seek(contentPos+int64(elemSize), io.SeekStart)
	}
	if isVideo || (width > 0 && height > 0) {
		return width, height
	}
	return 0, 0
}

func parseVideoSettings(rs io.ReadSeeker, endPos int64) (int, int) {
	var width, height int
	for {
		curr, err := rs.Seek(0, io.SeekCurrent)
		if err != nil || curr >= endPos {
			break
		}
		elemID, _, err := readVINT(rs, false)
		if err != nil {
			break
		}
		elemSize, _, err := readVINT(rs, true)
		if err != nil || elemSize == ebmlUnknownSize {
			break
		}
		contentPos, _ := rs.Seek(0, io.SeekCurrent)

		switch elemID {
		case idPixelWidth:
			if val, err := readUint(rs, elemSize); err == nil {
				width = int(val)
			}
		case idPixelHeight:
			if val, err := readUint(rs, elemSize); err == nil {
				height = int(val)
			}
		}
		rs.Seek(contentPos+int64(elemSize), io.SeekStart)
	}
	return width, height
}

// readVINT reads an EBML variable-length integer.
// If maskLengthBit is true, the leading length bit is masked off (used for data sizes).
// If maskLengthBit is false, the leading bit is retained (used for element IDs).
func readVINT(r io.Reader, maskLengthBit bool) (uint64, int, error) {
	var b [1]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, 0, err
	}
	first := b[0]
	if first == 0 {
		return 0, 0, errors.New("invalid vint: leading byte is zero")
	}

	length := 1
	mask := byte(0x80)
	for (first & mask) == 0 {
		mask >>= 1
		length++
	}

	var val uint64
	if maskLengthBit {
		val = uint64(first & (mask - 1))
	} else {
		val = uint64(first)
	}

	if length > 1 {
		rest := make([]byte, length-1)
		if _, err := io.ReadFull(r, rest); err != nil {
			return 0, 0, err
		}
		for i := 0; i < length-1; i++ {
			val = (val << 8) | uint64(rest[i])
		}
	}

	if maskLengthBit {
		// All data bits 1 signifies unknown size
		allOnes := (uint64(1) << (7 * length)) - 1
		if val == allOnes {
			return ebmlUnknownSize, length, nil
		}
	}

	return val, length, nil
}

func readUint(r io.Reader, size uint64) (uint64, error) {
	if size == 0 || size > 8 {
		return 0, fmt.Errorf("unsupported uint size: %d", size)
	}
	buf := make([]byte, size)
	if _, err := io.ReadFull(r, buf); err != nil {
		return 0, err
	}
	var val uint64
	for _, b := range buf {
		val = (val << 8) | uint64(b)
	}
	return val, nil
}

func readString(r io.Reader, size uint64) (string, error) {
	if size == 0 {
		return "", nil
	}
	if size > 64*1024*1024 {
		return "", fmt.Errorf("string size too large: %d", size)
	}
	buf := make([]byte, size)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return strings.TrimRight(string(buf), "\x00"), nil
}
