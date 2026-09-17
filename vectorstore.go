package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// ImageRecord 单张图片的向量记录
type ImageRecord struct {
	Path   string    `json:"path"`
	Vector []float32 `json:"vector"`
}

// VectorStore 向量存储（本地JSON持久化，生产环境替换为 Milvus / pgvector / qdrant）
type VectorStore struct {
	Records   []ImageRecord `json:"records"`
	ModelName string        `json:"model_name"` // 用于校验，防止模型混用
	Dim       int           `json:"dim"`        // 向量维度
}

// SearchResult 检索结果
type SearchResult struct {
	Path       string
	Similarity float32
}

// LoadVectorStore 从文件加载向量库
func LoadVectorStore(filePath string) (*VectorStore, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return &VectorStore{Records: []ImageRecord{}}, nil
		}
		return nil, err
	}
	var vs VectorStore
	if err := json.Unmarshal(data, &vs); err != nil {
		return nil, err
	}
	return &vs, nil
}

// Save 保存向量库到文件
func (vs *VectorStore) Save(filePath string) error {
	data, err := json.MarshalIndent(vs, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0644)
}

// Add 添加或更新一张图片
func (vs *VectorStore) Add(path string, vector []float32) {
	for i, r := range vs.Records {
		if r.Path == path {
			vs.Records[i].Vector = vector
			return
		}
	}
	vs.Records = append(vs.Records, ImageRecord{Path: path, Vector: vector})
}

// Search 余弦相似度检索，返回Top-K
func (vs *VectorStore) Search(query []float32, topK int) []SearchResult {
	results := make([]SearchResult, 0, len(vs.Records))
	for _, r := range vs.Records {
		sim := cosineSimilarity(query, r.Vector)
		results = append(results, SearchResult{Path: r.Path, Similarity: sim})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	if topK > len(results) {
		topK = len(results)
	}
	return results[:topK]
}

// Count 返回图片数量
func (vs *VectorStore) Count() int {
	return len(vs.Records)
}

// CheckModel 校验库的模型与当前配置是否一致，空库则初始化
func (vs *VectorStore) CheckModel(modelName string, dim int) error {
	if vs.ModelName == "" {
		vs.ModelName = modelName
		vs.Dim = dim
		return nil
	}
	if vs.ModelName != modelName {
		return fmt.Errorf("向量库模型不匹配：库是 %s，当前配置是 %s，请换存储文件或重建库", vs.ModelName, modelName)
	}
	return nil
}

// cosineSimilarity 余弦相似度
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
