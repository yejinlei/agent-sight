package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// EmbedProvider 嵌入服务提供商接口
type EmbedProvider interface {
	EmbedImage(imagePath string) ([]float32, error)
	ModelName() string
	Dim() int
}

// NewEmbedProvider 创建嵌入提供商
// 统一使用 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL 环境变量
// 兼容所有 OpenAI-compatible 服务（SiliconFlow / ModelArks / OpenAI 等）
func NewEmbedProvider() (EmbedProvider, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	baseURL := os.Getenv("OPENAI_BASE_URL")
	model := os.Getenv("OPENAI_MODEL")
	if apiKey == "" || baseURL == "" || model == "" {
		return nil, fmt.Errorf("缺少环境变量: OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL")
	}
	return &OpenAIProvider{apiKey: apiKey, baseURL: baseURL, model: model}, nil
}

// isFatalAPIError 判断是否为不可恢复的 API 硬错误
// 只有认证/配置类错误（401/403/404）会中止整个流程
// 400 通常只是单张图片参数问题，不该终止批处理
func isFatalAPIError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	fatalKeywords := []string{
		"401", "Unauthorized", "unauthorized",
		"403", "Forbidden", "forbidden",
		"404", "Not Found", "not found",
	}
	for _, k := range fatalKeywords {
		if strings.Contains(msg, k) {
			return true
		}
	}
	return false
}

// apiRetry 对瞬时错误（5xx/429/网络抖动）做指数退避重试
// 认证/配置类硬错误（401/403/404）和参数错误（400）立即返回，不浪费重试次数
func apiRetry(maxRetry int, fn func() error) error {
	var lastErr error
	backoff := 500 * time.Millisecond
	for attempt := 0; attempt <= maxRetry; attempt++ {
		if attempt > 0 {
			time.Sleep(backoff)
			backoff *= 2
			if backoff > 8000*time.Millisecond {
				backoff = 8 * time.Second
			}
			fmt.Fprintf(os.Stderr, "   ↻ 第%d次重试（%s后）... \n", attempt, backoff)
		}
		lastErr = fn()
		if lastErr == nil {
			return nil
		}
		fmt.Fprintf(os.Stderr, "   ⚠ 第%d次失败: %v\n", attempt+1, lastErr)
		if isFatalAPIError(lastErr) || isBadRequestError(lastErr) {
			return lastErr
		}
	}
	if maxRetry > 0 {
		return fmt.Errorf("重试 %d 次后仍失败: %w", maxRetry, lastErr)
	}
	return lastErr
}

// isBadRequestError 判断是否为参数错误（400）— 重试无用，直接返回
func isBadRequestError(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(err.Error(), "API返回错误 400") ||
		strings.Contains(err.Error(), "parameter is invalid")
}

// readImageDataURI 读取图片，必要时缩放/压缩，返回 data URI
// 完全使用 Go 标准库（image/png + image/jpeg + image/draw）实现，不依赖任何外部工具
// 环境变量：
//   - IMAGE_OPTIMIZE: compress（默认）| raw（不处理原样发送）
//   - IMAGE_MAX_SIDE: 最大边长像素（默认 1024）
//   - IMAGE_MAX_MB:   目标字节上限 MB（默认 2）
func readImageDataURI(imagePath string) (string, error) {
	maxSide := getenvInt("IMAGE_MAX_SIDE", 1024)
	maxBytes := getenvInt("IMAGE_MAX_MB", 2) * 1024 * 1024

	mode := strings.ToLower(strings.TrimSpace(os.Getenv("IMAGE_OPTIMIZE")))
	if mode == "raw" || mode == "off" || mode == "no" || mode == "0" || mode == "false" {
		return readImageRaw(imagePath)
	}

	imgBytes, err := ioutil.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("读取图片失败: %w", err)
	}

	// 尝试 Go 解码
	srcImg, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		// Go 无法解码（16-bit PNG、HEIF、AVIF、APNG 等）：
		// 无外部工具时可尝试直接发原字节（服务端可能能处理），但严格限制体积避免服务端硬拒
		if len(imgBytes) <= 4*1024*1024 {
			contentType := detectContentTypeFromBytes(imgBytes, imagePath)
			return fmt.Sprintf("data:%s;base64,%s", contentType, base64.StdEncoding.EncodeToString(imgBytes)), nil
		}
		return "", fmt.Errorf("Go 无法解码此图片(%v) 且体积过大(%.1fMB)，纯 Go 无法压缩，请转存为标准 8-bit PNG/JPG",
			err, float64(len(imgBytes))/(1024*1024))
	}

	b := srcImg.Bounds()
	// 已小于等于目标尺寸且体积可控 → 原样返回（省一次重编码）
	if b.Dx() <= maxSide && b.Dy() <= maxSide && len(imgBytes) <= maxBytes {
		contentType := detectImageContentType(imagePath)
		return fmt.Sprintf("data:%s;base64,%s", contentType, base64.StdEncoding.EncodeToString(imgBytes)), nil
	}

	// 居中裁剪成方形（视觉 embedding 模型普遍偏好吃方形输入）
	side := b.Dx()
	if b.Dy() < side {
		side = b.Dy()
	}
	top := b.Min.Y + (b.Dy()-side)/2
	left := b.Min.X + (b.Dx()-side)/2
	cropped := image.NewRGBA(image.Rect(0, 0, side, side))
	draw.Draw(cropped, cropped.Bounds(), srcImg, image.Point{X: left, Y: top}, draw.Src)

	// 缩放到 maxSide × maxSide
	var final image.Image = cropped
	if side != maxSide {
		final = scaleImage(cropped, maxSide, maxSide)
	}

	// 优先 JPEG 编码（对照片类图效果好、体积小）；失败或超阈值时降级为 PNG
	buf := encodeJPEGToSize(final, maxBytes)
	if buf != nil {
		return fmt.Sprintf("data:image/jpeg;base64,%s", base64.StdEncoding.EncodeToString(buf)), nil
	}

	// PNG 兜底：BestCompression 无损压缩
	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, final); err != nil {
		return "", fmt.Errorf("编码失败: %w", err)
	}
	// PNG 体积通常大于 JPEG，若仍超过阈值也照发（避免 API 端解析失败）
	return fmt.Sprintf("data:image/png;base64,%s", base64.StdEncoding.EncodeToString(pngBuf.Bytes())), nil
}

// encodeJPEGToSize 尝试多种质量等级编码 JPEG，返回体积最小的合规结果
func encodeJPEGToSize(img image.Image, maxBytes int) []byte {
	var best []byte
	for _, q := range []int{90, 82, 72, 62, 52, 42} {
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: q}); err != nil {
			continue
		}
		if best == nil || buf.Len() < len(best) {
			best = append([]byte(nil), buf.Bytes()...)
		}
		if buf.Len() <= maxBytes {
			return buf.Bytes()
		}
	}
	return best
}

// readImageRaw 原样返回 data URI，不做任何解码/缩放/压缩
func readImageRaw(imagePath string) (string, error) {
	imgBytes, err := ioutil.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("读取图片失败: %w", err)
	}
	contentType := detectImageContentType(imagePath)
	return fmt.Sprintf("data:%s;base64,%s", contentType, base64.StdEncoding.EncodeToString(imgBytes)), nil
}

// scaleImage 双线性缩放
func scaleImage(src image.Image, w, h int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	xscale := float64(src.Bounds().Dx()) / float64(w)
	yscale := float64(src.Bounds().Dy()) / float64(h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dst.Set(x, y, src.At(int(float64(x)*xscale)+src.Bounds().Min.X, int(float64(y)*yscale)+src.Bounds().Min.Y))
		}
	}
	return dst
}

func detectImageContentType(path string) string {
	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".png"):
		return "image/png"
	case strings.HasSuffix(lower, ".webp"):
		return "image/webp"
	case strings.HasSuffix(lower, ".gif"):
		return "image/gif"
	default:
		return "image/jpeg"
	}
}

// detectContentTypeFromBytes 通过文件头魔数检测真实 MIME 类型
func detectContentTypeFromBytes(b []byte, path string) string {
	if len(b) >= 4 {
		if b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF {
			return "image/jpeg"
		}
		if b[0] == 0x89 && b[1] == 0x50 && b[2] == 0x4E && b[3] == 0x47 {
			return "image/png"
		}
		if b[0] == 'G' && b[1] == 'I' && b[2] == 'F' && b[3] == '8' {
			return "image/gif"
		}
		if len(b) >= 12 && b[0] == 'R' && b[1] == 'I' && b[2] == 'F' && b[3] == 'F' &&
			b[8] == 'W' && b[9] == 'E' && b[10] == 'B' && b[11] == 'P' {
			return "image/webp"
		}
		if len(b) >= 16 && b[4] == 'f' && b[5] == 't' && b[6] == 'y' && b[7] == 'p' {
			return "image/heif"
		}
	}
	return detectImageContentType(path)
}

func getenvInt(key string, def int) int {
	s := strings.TrimSpace(os.Getenv(key))
	if s == "" {
		return def
	}
	if n, err := strconv.Atoi(s); err == nil && n > 0 {
		return n
	}
	return def
}

// ---------- OpenAI 兼容提供商 ----------
type OpenAIProvider struct {
	apiKey  string
	baseURL string
	model   string
}

type openAIEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (o *OpenAIProvider) ModelName() string { return o.model }

func (o *OpenAIProvider) Dim() int {
	switch o.model {
	case "PaddlePaddle/ernie_vil-2.0-base-zh", "openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32":
		return 768
	default:
		return 1024
	}
}

func (o *OpenAIProvider) EmbedImage(imagePath string) ([]float32, error) {
	var result openAIEmbedResponse
	err := apiRetry(3, func() error {
		dataURI, err := readImageDataURI(imagePath)
		if err != nil {
			return err
		}
		return o.doEmbed(dataURI, &result)
	})
	if err != nil {
		return nil, err
	}
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("未返回嵌入向量")
	}
	return result.Data[0].Embedding, nil
}

func (o *OpenAIProvider) doEmbed(dataURI string, out *openAIEmbedResponse) error {
	// 不同供应商对视觉 embedding 的请求格式不同：
	//   - SiliconFlow Qwen3-VL-Embedding-8B 要求 {"input":{"image":"..."}}（视觉专用 schema）
	//   - 部分网关（如 Moark）走 OpenAI 兼容，input 直接放字符串
	// 我们按"先视觉格式、失败降级为文本格式"的顺序尝试，对两种 schema 都能兼容。
	variants := []interface{}{dataURI}
	if strings.HasPrefix(dataURI, "data:image/") {
		variants = []interface{}{map[string]string{"image": dataURI}, dataURI}
	}
	var lastErr error
	for _, inputVal := range variants {
		err := o.doEmbedOnce(inputVal, out)
		if err == nil {
			return nil
		}
		lastErr = err
		// 只对 400 类错误降级重试，其他错误直接返回
		if !strings.Contains(err.Error(), "API返回错误 400") {
			return err
		}
	}
	return lastErr
}

func (o *OpenAIProvider) doEmbedOnce(inputVal interface{}, out *openAIEmbedResponse) error {
	reqBody := map[string]interface{}{
		"model":           o.model,
		"input":           inputVal,
		"encoding_format": "float",
	}
	bodyBytes, _ := json.Marshal(reqBody)
	req, err := http.NewRequest("POST", o.baseURL+"/embeddings", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("API请求失败: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := ioutil.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return fmt.Errorf("API返回错误 %d: %s", resp.StatusCode, string(respBody))
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("解析响应失败: %w", err)
	}
	return nil
}
