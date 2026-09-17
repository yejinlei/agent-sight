package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Reranker VLM 精排器
type Reranker struct {
	apiKey  string
	baseURL string
	model   string
}

// NewReranker 创建VLM精排器，复用 OPENAI_API_KEY / OPENAI_BASE_URL 网关
func NewReranker() *Reranker {
	apiKey := os.Getenv("OPENAI_API_KEY")
	baseURL := os.Getenv("OPENAI_BASE_URL")
	model := os.Getenv("VLM_MODEL")
	if model == "" {
		model = "Qwen/Qwen2-VL-72B-Instruct"
	}
	return &Reranker{apiKey: apiKey, baseURL: baseURL, model: model}
}

// RerankResult VLM 精排结果
type RerankResult struct {
	Path    string  `json:"path"`
	Score   float64 `json:"score"` // 0-10 分
	Reason  string  `json:"reason"`
	IsMatch bool    `json:"is_match"`
}

// imageToDataURI 图片转 base64 data URI
func imageToDataURI(path string) (string, error) {
	imgBytes, err := ioutil.ReadFile(path)
	if err != nil {
		return "", err
	}
	contentType := "image/jpeg"
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".png" {
		contentType = "image/png"
	} else if ext == ".webp" {
		contentType = "image/webp"
	}
	return fmt.Sprintf("data:%s;base64,%s", contentType, base64.StdEncoding.EncodeToString(imgBytes)), nil
}

// Rerank 调用VLM对候选图进行精排
// queryPath: 查询图片路径
// candidates: 嵌入检索返回的候选列表（已按相似度降序）
// textQuery: 可选文字筛选条件，空字符串表示纯视觉比对
// 内部会截取 VLM_BATCH_SIZE 张最相似的图送 VLM，避免一次请求体过大
func (r *Reranker) Rerank(queryPath string, candidates []SearchResult, textQuery string) ([]RerankResult, error) {
	batchSize := GetVLMBatchSize()
	if batchSize > 0 && len(candidates) > batchSize {
		candidates = candidates[:batchSize]
	}

	queryURI, err := imageToDataURI(queryPath)
	if err != nil {
		return nil, fmt.Errorf("读取查询图失败: %w", err)
	}

	systemPrompt := `你是专业的图像相似度比对专家。我会给你一张【查询图片】和多张【候选图片】，请你逐一比对候选图片与查询图片的相似程度。

评分标准（0-10分）：
- 9-10分：完全相同或几乎完全相同（同一主体、同一场景、同一拍摄角度，只是分辨率/裁剪不同）
- 7-8分：高度相似（同一主体、同一场景，拍摄角度/光线有差异）
- 5-6分：一般相似（主体相同，但场景/姿态不同，或视觉风格相似）
- 3-4分：有点像，但明显不是同一事物（只是颜色或构图类似）
- 0-2分：完全不相关

严格按照以下JSON格式返回结果（不要输出其他任何解释文字）：
{
  "results": [
    {"index": 1, "score": 分数, "reason": "简短说明匹配原因", "is_match": true/false},
    ...
  ]
}
其中 index 是候选图片编号，从 1 开始；is_match 表示是否与查询图是同一事物。`

	userPrompt := ""
	if textQuery != "" {
		userPrompt += fmt.Sprintf("【额外筛选条件】除了视觉相似，还必须满足：%s。不满足条件的图 is_match 设为 false。\n\n", textQuery)
	}
	userPrompt += "候选图片按顺序编号为 候选1、候选2、...，现在开始："

	// 构造多模态消息内容
	content := []map[string]interface{}{
		{"type": "text", "text": systemPrompt + "\n\n【查询图片】："},
		{"type": "image_url", "image_url": map[string]string{"url": queryURI, "detail": "low"}},
		{"type": "text", "text": "\n\n" + userPrompt + "\n"},
	}

	// 加入所有候选图，按编号标识
	for i, c := range candidates {
		candidateURI, err := imageToDataURI(c.Path)
		if err != nil {
			continue
		}
		content = append(content, map[string]interface{}{
			"type": "text", "text": fmt.Sprintf("\n候选%d：", i+1),
		})
		content = append(content, map[string]interface{}{
			"type": "image_url", "image_url": map[string]string{"url": candidateURI, "detail": "low"},
		})
	}

	content = append(content, map[string]interface{}{
		"type": "text", "text": "\n\n请返回JSON：",
	})

	reqBody := map[string]interface{}{
		"model": r.model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": content},
		},
		"max_tokens":  4096,
		"temperature": 0.1,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	req, err := http.NewRequest("POST", r.baseURL+"/chat/completions", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+r.apiKey)

	client := &http.Client{Timeout: time.Duration(GetVLMTimeoutSeconds()) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("VLM API请求失败: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := ioutil.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("VLM API错误 %d: %s", resp.StatusCode, string(respBody))
	}

	// 解析响应
	var respData struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &respData); err != nil {
		return nil, fmt.Errorf("解析VLM响应失败: %w, 响应: %s", err, string(respBody))
	}
	if len(respData.Choices) == 0 {
		return nil, fmt.Errorf("VLM未返回结果")
	}

	resultText := respData.Choices[0].Message.Content
	// 去掉可能的markdown代码块标记
	resultText = strings.TrimSpace(resultText)
	resultText = strings.TrimPrefix(resultText, "```json")
	resultText = strings.TrimPrefix(resultText, "```")
	resultText = strings.TrimSuffix(resultText, "```")
	resultText = strings.TrimSpace(resultText)

	// 解析VLM返回的评分
	type vlmItem struct {
		Index   int     `json:"index"`
		Score   float64 `json:"score"`
		Reason  string  `json:"reason"`
		IsMatch bool    `json:"is_match"`
	}
	var wrapper struct {
		Results []vlmItem `json:"results"`
	}
	if err := json.Unmarshal([]byte(resultText), &wrapper); err != nil {
		// 尝试直接解析数组
		var arr []vlmItem
		if err2 := json.Unmarshal([]byte(resultText), &arr); err2 != nil {
			return nil, fmt.Errorf("解析VLM评分JSON失败: %w, 内容: %s", err, resultText)
		}
		wrapper.Results = arr
	}

	// 映射回原路径
	results := make([]RerankResult, 0, len(candidates))
	for _, vr := range wrapper.Results {
		idx := vr.Index - 1
		if idx < 0 || idx >= len(candidates) {
			continue
		}
		results = append(results, RerankResult{
			Path:    candidates[idx].Path,
			Score:   vr.Score,
			Reason:  vr.Reason,
			IsMatch: vr.IsMatch,
		})
	}

	// 按VLM评分降序排序
	for i := range results {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	return results, nil
}

// GetAutoRerankThreshold 获取自动精排阈值（数字形式，兼容旧调用点）
func GetAutoRerankThreshold() float64 {
	val := GetAutoRerankThresholdString()
	f, err := strconv.ParseFloat(val, 64)
	if err != nil {
		return 0.85
	}
	return f
}

// GetAutoRerankThresholdString 返回原始阈值字符串，支持 'auto' 关键字。
// 若未设置或为空，默认返回 "auto"。
func GetAutoRerankThresholdString() string {
	val := strings.TrimSpace(os.Getenv("AUTO_RERANK_THRESHOLD"))
	if val == "" {
		return "auto"
	}
	return val
}

// GetRecallCandidates 获取粗召回数量
func GetRecallCandidates() int {
	val := os.Getenv("RECALL_CANDIDATES")
	if val == "" {
		return 50
	}
	n, err := strconv.Atoi(val)
	if err != nil {
		return 50
	}
	return n
}

// GetVLMBatchSize 获取单次送 VLM 精排的候选图数量
// 从 RECALL_CANDIDATES 召回后，只把最相似的 N 张真正发给 VLM
// 默认 6，可通过 VLM_BATCH_SIZE 环境变量调整
func GetVLMBatchSize() int {
	val := os.Getenv("VLM_BATCH_SIZE")
	if val == "" {
		return 6
	}
	n, err := strconv.Atoi(val)
	if err != nil {
		return 6
	}
	return n
}

// GetVLMTimeoutSeconds 获取 VLM 请求超时秒数
// 默认 120s，超时后客户端主动断开，由上层降级返回嵌入结果
func GetVLMTimeoutSeconds() int {
	val := os.Getenv("VLM_TIMEOUT")
	if val == "" {
		return 120
	}
	n, err := strconv.Atoi(val)
	if err != nil {
		return 120
	}
	if n <= 0 {
		return 120
	}
	return n
}
