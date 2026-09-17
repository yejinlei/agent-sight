package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// WebServer 本地 Web 服务，复用现有 buildLibrary 逻辑
type WebServer struct {
	storeFile string
	buildMu   sync.Mutex
	buildBusy bool
}

func NewWebServer(storeFile string) *WebServer {
	return &WebServer{storeFile: storeFile}
}

// Serve 启动 HTTP 服务并自动打开浏览器
func (ws *WebServer) Serve(addr string) error {
	mux := http.NewServeMux()

	mux.HandleFunc("/", ws.handleIndex)
	mux.HandleFunc("/api/info", ws.handleInfo)
	mux.HandleFunc("/api/build", ws.handleBuild)
	mux.HandleFunc("/api/search", ws.handleSearch)
	mux.HandleFunc("/api/search-pro", ws.handleSearchPro)
	mux.HandleFunc("/img", ws.handleImage)

	srv := &http.Server{Addr: addr, Handler: mux}

	// 延迟 300ms 打开浏览器，让监听起来
	go func() {
		time.Sleep(300 * time.Millisecond)
		openBrowser("http://" + addr)
	}()

	fmt.Printf("🌐 Web 服务已启动: http://%s\n", addr)
	fmt.Printf("   按 Ctrl+C 退出\n")
	return srv.ListenAndServe()
}

// openBrowser 跨平台打开默认浏览器
func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	if err := cmd.Start(); err != nil {
		log.Printf("⚠️  无法自动打开浏览器，请手动访问 %s", url)
	}
}

// ---------- Handlers ----------

func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = io.WriteString(w, indexHTML)
}

func (ws *WebServer) handleInfo(w http.ResponseWriter, r *http.Request) {
	info := map[string]interface{}{
		"count":       0,
		"model":       "",
		"dim":         0,
		"error":       "",
		"topKDefault": topKDefault(),
		"threshold":   GetAutoRerankThresholdString(),
	}
	vs, err := LoadVectorStore(ws.storeFile)
	if err != nil {
		info["error"] = err.Error()
	} else {
		info["count"] = vs.Count()
		info["model"] = vs.ModelName
		info["dim"] = vs.Dim
	}
	_ = json.NewEncoder(w).Encode(info)
}

type buildReq struct {
	Dir   string `json:"dir"`
	Force bool   `json:"force"`
}

func (ws *WebServer) handleBuild(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}
	ws.buildMu.Lock()
	if ws.buildBusy {
		ws.buildMu.Unlock()
		http.Error(w, "build 正在执行中", 409)
		return
	}
	ws.buildBusy = true
	ws.buildMu.Unlock()
	defer func() {
		ws.buildMu.Lock()
		ws.buildBusy = false
		ws.buildMu.Unlock()
	}()

	var req buildReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), 400)
		return
	}
	req.Dir = strings.TrimSpace(req.Dir)
	if req.Dir == "" {
		http.Error(w, "dir 不能为空", 400)
		return
	}
	if _, err := os.Stat(req.Dir); err != nil {
		http.Error(w, "目录不存在: "+err.Error(), 400)
		return
	}

	buildLibrary(req.Dir, ws.storeFile, 3, req.Force)

	vs, err := LoadVectorStore(ws.storeFile)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":    true,
		"count": vs.Count(),
	})
}

// topKDefault 从 .env 读 SEARCH_TOP_K 作为 Web 默认返回数量；空或未设置时返回 5
func topKDefault() int {
	val := strings.TrimSpace(os.Getenv("SEARCH_TOP_K"))
	if val == "" || strings.EqualFold(val, "auto") {
		return 5
	}
	n, err := strconv.Atoi(val)
	if err != nil || n <= 0 || n > 50 {
		return 5
	}
	return n
}

// uploadTemp 保存 multipart 上传的图片到临时文件
func (ws *WebServer) uploadTemp(w http.ResponseWriter, r *http.Request, field string) (string, error) {
	if err := r.ParseMultipartForm(20 << 20); err != nil {
		return "", err
	}
	file, header, err := r.FormFile(field)
	if err != nil {
		return "", err
	}
	defer file.Close()

	ext := filepath.Ext(header.Filename)
	tmpPath := filepath.Join(os.TempDir(), fmt.Sprintf("agent-sight-%d-%d%s", os.Getpid(), time.Now().UnixNano(), ext))
	out, err := os.Create(tmpPath)
	if err != nil {
		return "", err
	}
	defer out.Close()
	if _, err := io.Copy(out, file); err != nil {
		return "", err
	}
	return tmpPath, nil
}

func (ws *WebServer) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}
	qPath, err := ws.uploadTemp(w, r, "image")
	if err != nil {
		http.Error(w, "上传图片失败: "+err.Error(), 400)
		return
	}
	defer os.Remove(qPath)

	topK := resolveTopK(r.FormValue("topK"), 5)

	results, info := ws.runSearch(qPath, topK)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":      true,
		"results": results,
		"auto":    info,
	})
}

// resolveTopK 解析 topK 参数：
//   - ""/auto/非法 → fallback
//   - "auto" → 返回 0，让 runSearch 走自适应筛选
//   - 数字 → 直接返回（1..100）
func resolveTopK(raw string, fallback int) int {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fallback
	}
	if strings.EqualFold(raw, "auto") {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return fallback
	}
	if n > 100 {
		return 100
	}
	return n
}

// parseTopKParam 处理 CLI 风格的 topK 参数，返回 (值, 是否 auto)。
func parseTopKParam(raw string, fallback int) (int, bool) {
	n := resolveTopK(raw, fallback)
	return n, n == 0
}

func (ws *WebServer) handleSearchPro(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}
	qPath, err := ws.uploadTemp(w, r, "image")
	if err != nil {
		http.Error(w, "上传图片失败: "+err.Error(), 400)
		return
	}
	defer os.Remove(qPath)

	textQuery := strings.TrimSpace(r.FormValue("text"))
	topK := 5
	if tk := r.FormValue("topK"); tk != "" {
		if n, err := strconv.Atoi(tk); err == nil && n > 0 {
			topK = n
		}
	}

	results, reranked := ws.runSearchPro(qPath, textQuery, topK)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":       true,
		"results":  results,
		"reranked": reranked,
	})
}

// runSearch 快速检索（纯向量）
// 返回 (结果列表, auto 信息)。auto 信息包含判定阈值、Top-1 分数、是否建议精排等。
func (ws *WebServer) runSearch(queryPath string, topK int) ([]map[string]interface{}, map[string]interface{}) {
	vs, err := LoadVectorStore(ws.storeFile)
	if err != nil || vs.Count() == 0 {
		return nil, map[string]interface{}{"ok": false, "reason": "向量库为空"}
	}
	provider, err := NewEmbedProvider()
	if err != nil {
		return nil, map[string]interface{}{"ok": false, "reason": "嵌入服务初始化失败: " + err.Error()}
	}
	vec, err := provider.EmbedImage(queryPath)
	if err != nil {
		return nil, map[string]interface{}{"ok": false, "reason": "查询图向量化失败: " + err.Error()}
	}

	// 取全库候选，稍后再按 topK / auto 阈值裁剪
	all := vs.Search(vec, vs.Count())
	if len(all) == 0 {
		return nil, map[string]interface{}{"ok": false, "reason": "无匹配结果"}
	}

	// 阈值判定（auto 或固定）
	top1 := float64(all[0].Similarity)
	thresholdStr := strings.TrimSpace(GetAutoRerankThresholdString())
	autoMode := strings.EqualFold(thresholdStr, "auto")
	info := map[string]interface{}{
		"ok":         true,
		"mode":       "auto",
		"top1":       top1,
		"suggestPro": false,
		"topK":       topK,
		"auto":       autoMode,
	}

	// 判定阈值 & 结果条数
	keepN := topK // 默认按 topK 裁剪
	if topK <= 0 {
		keepN = len(all) // auto 模式先全量保留，按阈值筛
	}

	if autoMode {
		p90, distN := computeAutoThreshold(vs)
		info["p90"] = p90
		info["distN"] = distN
		if distN == 0 {
			info["reason"] = "库太小，跳过 auto 判定"
		} else {
			gap := top1 - float64(all[1].Similarity)
			if top1 < p90 || (gap < 0.05 && top1 < p90+0.05) {
				info["suggestPro"] = true
				info["reason"] = fmt.Sprintf("库内分布 p90=%.4f, Top-1=%.4f (差 %.4f) 匹配不足", p90, top1, gap)
			} else {
				info["reason"] = fmt.Sprintf("库内分布 p90=%.4f, Top-1=%.4f 匹配充分", p90, top1)
			}
		}
		// auto 模式：按阈值自动筛选返回数量
		if topK <= 0 {
			cut := p90
			keepN = 0
			for _, r := range all {
				if float64(r.Similarity) >= cut {
					keepN++
				} else {
					break
				}
			}
			// 至少返回 1 条；最多 50 条
			if keepN < 1 {
				keepN = 1
			}
			if keepN > 50 {
				keepN = 50
			}
			info["autoKeep"] = keepN
		}
	} else {
		thr := GetAutoRerankThreshold()
		info["mode"] = "fixed"
		info["threshold"] = thr
		if top1 < thr {
			info["suggestPro"] = true
			info["reason"] = fmt.Sprintf("Top-1=%.4f 低于固定阈值 %.4f", top1, thr)
		} else {
			info["reason"] = fmt.Sprintf("Top-1=%.4f 达到固定阈值 %.4f", top1, thr)
		}
		// fixed 模式 + auto 返回数量：按阈值筛
		if topK <= 0 {
			keepN = 0
			for _, r := range all {
				if float64(r.Similarity) >= thr {
					keepN++
				} else {
					break
				}
			}
			if keepN < 1 {
				keepN = 1
			}
			if keepN > 50 {
				keepN = 50
			}
			info["autoKeep"] = keepN
		}
	}

	if keepN > len(all) {
		keepN = len(all)
	}
	results := all[:keepN]
	info["returned"] = len(results)

	out := make([]map[string]interface{}, 0, len(results))
	for _, r := range results {
		out = append(out, map[string]interface{}{
			"path":       r.Path,
			"similarity": float64(r.Similarity),
		})
	}

	return out, info
}

// runSearchPro 精准检索（嵌入召回 + VLM 精排）
func (ws *WebServer) runSearchPro(queryPath, textQuery string, topK int) ([]map[string]interface{}, bool) {
	vs, err := LoadVectorStore(ws.storeFile)
	if err != nil || vs.Count() == 0 {
		return nil, false
	}
	provider, err := NewEmbedProvider()
	if err != nil {
		return nil, false
	}
	vec, err := provider.EmbedImage(queryPath)
	if err != nil {
		return nil, false
	}

	recallN := GetRecallCandidates()
	if recallN > vs.Count() {
		recallN = vs.Count()
	}
	recall := vs.Search(vec, recallN)

	reranker := NewReranker()
	rankings, err := reranker.Rerank(queryPath, recall, textQuery)
	if err != nil || len(rankings) == 0 {
		out := make([]map[string]interface{}, 0, topK)
		for i, r := range recall {
			if i >= topK {
				break
			}
			out = append(out, map[string]interface{}{
				"path":       r.Path,
				"similarity": float64(r.Similarity),
				"score":      nil,
			})
		}
		return out, false
	}

	out := make([]map[string]interface{}, 0, topK)
	for i, r := range rankings {
		if i >= topK {
			break
		}
		out = append(out, map[string]interface{}{
			"path":     r.Path,
			"score":    r.Score,
			"reason":   r.Reason,
			"is_match": r.IsMatch,
		})
	}
	return out, true
}

// handleImage 提供图库图片访问，防止目录穿越
func (ws *WebServer) handleImage(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("path")
	if path == "" {
		http.Error(w, "missing path", 400)
		return
	}
	absTarget, err := filepath.Abs(path)
	if err != nil {
		http.Error(w, "bad path", 400)
		return
	}
	absCwd, err := filepath.Abs(".")
	if err != nil {
		http.Error(w, "internal error", 500)
		return
	}
	if !strings.HasPrefix(absTarget, absCwd+string(filepath.Separator)) && absTarget != absCwd {
		http.Error(w, "forbidden", 403)
		return
	}
	if _, err := os.Stat(absTarget); err != nil {
		http.Error(w, "not found", 404)
		return
	}
	ext := strings.ToLower(filepath.Ext(absTarget))
	var ct string
	switch ext {
	case ".jpg", ".jpeg":
		ct = "image/jpeg"
	case ".png":
		ct = "image/png"
	case ".gif":
		ct = "image/gif"
	case ".webp":
		ct = "image/webp"
	case ".bmp":
		ct = "image/bmp"
	case ".svg":
		ct = "image/svg+xml"
	default:
		ct = "application/octet-stream"
	}
	w.Header().Set("Content-Type", ct)
	http.ServeFile(w, r, absTarget)
}

// ---------- HTML ----------

var indexHTML = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>agent-sight · 多模态检索</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: #f7f7f8; color: #1a1a1a; height: 100vh; overflow: hidden; display: flex; }
.sidebar { width: 240px; background: #fff; border-right: 1px solid #eaeaea; padding: 20px 16px; display: flex; flex-direction: column; gap: 8px; }
.logo { font-size: 16px; font-weight: 700; padding: 8px 12px; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.logo-dot { width: 10px; height: 10px; background: linear-gradient(135deg,#3b82f6,#8b5cf6); border-radius: 3px; }
.nav-item { padding: 10px 12px; border-radius: 8px; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 10px; color: #333; }
.nav-item:hover { background: #f0f0f2; }
.nav-item.active { background: #e8f0ff; color: #1a73e8; font-weight: 600; }
.sidebar-foot { margin-top: auto; font-size: 11px; color: #999; padding: 12px; line-height: 1.6; }
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.header { padding: 16px 32px; border-bottom: 1px solid #eaeaea; background: #fff; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 18px; font-weight: 600; }
.header .stats { font-size: 12px; color: #666; }
.content { flex: 1; overflow-y: auto; padding: 24px 32px; }
.card { background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.card h2 { font-size: 15px; margin-bottom: 12px; font-weight: 600; }
.card .desc { font-size: 12px; color: #888; margin-bottom: 16px; line-height: 1.6; }
.row { display: flex; gap: 12px; align-items: flex-end; margin-bottom: 12px; }
.field { display: flex; flex-direction: column; gap: 6px; flex: 1; }
.field label { font-size: 12px; color: #555; font-weight: 500; }
.field input, .field textarea { padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px; font-family: inherit; }
.field input:focus, .field textarea:focus { outline: none; border-color: #3b82f6; }
.btn { padding: 8px 16px; background: #1a73e8; color: #fff; border: none; border-radius: 6px; font-size: 13px; cursor: pointer; font-weight: 500; }
.btn:hover { background: #1557b0; }
.btn:disabled { background: #ccc; cursor: not-allowed; }
.upload { border: 2px dashed #d1d5db; border-radius: 8px; padding: 24px; text-align: center; cursor: pointer; background: #fafafa; }
.upload:hover { border-color: #3b82f6; background: #f0f5ff; }
.upload-preview { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
.thumb { width: 100px; height: 100px; border: 1px solid #e5e7eb; border-radius: 8px; object-fit: cover; }
.results { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 16px; margin-top: 16px; }
.result-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; transition: transform 0.15s; }
.result-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.result-card img { width: 100%; height: 160px; object-fit: cover; display: block; background: #f5f5f5; }
.result-card .info { padding: 8px 10px; }
.result-card .path { font-size: 11px; color: #666; word-break: break-all; margin-bottom: 4px; }
.result-card .score { font-size: 12px; font-weight: 600; color: #1a73e8; }
.result-card .rank { display: inline-block; background: #1a73e8; color: #fff; font-size: 11px; padding: 2px 6px; border-radius: 10px; margin-right: 6px; }
.loading { padding: 20px; text-align: center; color: #666; font-size: 13px; }
.error { padding: 12px; background: #fef2f2; color: #dc2626; border-radius: 6px; font-size: 13px; margin-top: 12px; }
.auto-banner { margin-top: 12px; padding: 12px 14px; border-radius: 6px; background: #f0fdf4; border: 1px solid #bbf7d0; font-size: 13px; }
.auto-banner.warn { background: #fffbeb; border-color: #fcd34d; }
.auto-banner.error { background: #fef2f2; border-color: #fca5a5; }
.auto-banner .banner-row { display: flex; align-items: center; gap: 10px; }
.auto-banner .banner-icon { font-size: 18px; flex-shrink: 0; }
.auto-banner .banner-body { flex: 1; min-width: 0; }
.auto-banner .banner-title { font-weight: 600; color: #166534; }
.auto-banner.warn .banner-title { color: #92400e; }
.auto-banner.error .banner-title { color: #991b1b; }
.auto-banner .banner-detail { color: #555; font-size: 12px; margin-top: 3px; word-break: break-all; }
.auto-banner code { background: rgba(0,0,0,0.06); padding: 1px 5px; border-radius: 3px; font-size: 11px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; background: #e0f2e9; color: #059669; margin-left: 8px; }
.hidden { display: none; }
</style>
</head>
<body>
<aside class="sidebar">
  <div class="logo"><div class="logo-dot"></div>agent-sight</div>
  <div class="nav-item active" data-tab="search">🔍 快速检索</div>
  <div class="nav-item" data-tab="searchpro">✨ 精准检索</div>
  <div class="nav-item" data-tab="build">📥 建库</div>
  <div class="nav-item" data-tab="info">ℹ️ 图库信息</div>
  <div class="sidebar-foot">
    Web 控制台<br>
    <span id="model-info">加载中...</span>
  </div>
</aside>

<main class="main">
  <header class="header">
    <h1 id="tab-title">快速检索</h1>
    <div class="stats" id="stats">图库: 加载中...</div>
  </header>
  <div class="content" id="content">
    <div class="tab-panel" id="panel-search">
      <div class="card">
        <h2>上传查询图片</h2>
        <div class="desc">上传图片后与图库中所有图片进行向量相似度比对，返回最相似的 Top-K 张。</div>
        <div class="upload" id="search-upload">点击选择图片，或拖拽到此处</div>
        <input type="file" id="search-file" accept="image/*" class="hidden">
        <div class="upload-preview" id="search-preview"></div>
        <div class="row" style="margin-top:16px;">
          <div class="field" style="max-width:140px;">
            <label>返回数量</label>
            <select id="search-topk"></select>
          </div>
          <button class="btn" id="search-btn">开始检索</button>
        </div>
        <div class="error hidden" id="search-error"></div>
        <div class="auto-banner hidden" id="search-auto"></div>
        <div class="loading hidden" id="search-loading">检索中...</div>
        <div class="results" id="search-results"></div>
      </div>
    </div>

    <div class="tab-panel hidden" id="panel-searchpro">
      <div class="card">
        <h2>精准检索 <span class="badge">VLM 精排</span></h2>
        <div class="desc">先用向量召回候选图，再送 VLM 大模型基于图片和文字条件做语义精排。</div>
        <div class="upload" id="pro-upload">点击选择图片，或拖拽到此处</div>
        <input type="file" id="pro-file" accept="image/*" class="hidden">
        <div class="upload-preview" id="pro-preview"></div>
        <div class="row" style="margin-top:16px;">
          <div class="field">
            <label>文字筛选条件（可选）</label>
            <input type="text" id="pro-text" placeholder="如：人物是杨幂 / 有人笑 / 红色衣服">
          </div>
          <div class="field" style="max-width:140px;">
            <label>返回数量</label>
            <select id="pro-topk"></select>
          </div>
          <button class="btn" id="pro-btn">开始检索</button>
        </div>
        <div class="error hidden" id="pro-error"></div>
        <div class="loading hidden" id="pro-loading">VLM 精排中（约 5-30 秒）...</div>
        <div class="results" id="pro-results"></div>
      </div>
    </div>

    <div class="tab-panel hidden" id="panel-build">
      <div class="card">
        <h2>建库</h2>
        <div class="desc">输入图片目录路径，将目录下的所有图片做向量化入库。已入库的默认跳过，勾选"强制"可重建。</div>
        <div class="row">
          <div class="field">
            <label>目录路径（相对或绝对）</label>
            <input type="text" id="build-dir" placeholder="./input 或 /full/path">
          </div>
        </div>
        <div class="row">
          <label style="display:flex;align-items:center;gap:6px;font-size:13px;color:#555;">
            <input type="checkbox" id="build-force"> 强制重建（覆盖已有）
          </label>
          <button class="btn" id="build-btn">开始建库</button>
        </div>
        <div class="error hidden" id="build-error"></div>
        <div class="loading hidden" id="build-loading">建库中（每张约 1-3 秒）...</div>
        <div id="build-success" class="hidden" style="padding:12px;background:#e0f2e9;color:#059669;border-radius:6px;font-size:13px;margin-top:12px;"></div>
      </div>
    </div>

    <div class="tab-panel hidden" id="panel-info">
      <div class="card">
        <h2>图库信息</h2>
        <div id="info-content">加载中...</div>
      </div>
    </div>
  </div>
</main>

<script>
const tabs = {
  search: '快速检索',
  searchpro: '精准检索',
  build: '建库',
  info: '图库信息'
};

document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', () => {
    const tab = el.dataset.tab;
    document.querySelectorAll('.nav-item').forEach(x => x.classList.remove('active'));
    el.classList.add('active');
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    document.getElementById('panel-' + tab).classList.remove('hidden');
    document.getElementById('tab-title').textContent = tabs[tab];
    if (tab === 'info') loadInfo();
  });
});

function setupUpload(uploadId, fileId, previewId) {
  const upload = document.getElementById(uploadId);
  const input = document.getElementById(fileId);
  const preview = document.getElementById(previewId);
  upload.addEventListener('click', () => input.click());
  upload.addEventListener('dragover', e => { e.preventDefault(); upload.style.borderColor = '#3b82f6'; });
  upload.addEventListener('dragleave', () => upload.style.borderColor = '#d1d5db');
  upload.addEventListener('drop', e => {
    e.preventDefault();
    upload.style.borderColor = '#d1d5db';
    if (e.dataTransfer.files[0]) {
      input.files = e.dataTransfer.files;
      input.dispatchEvent(new Event('change'));
    }
  });
  input.addEventListener('change', () => {
    if (input.files[0]) {
      const url = URL.createObjectURL(input.files[0]);
      preview.innerHTML = '<img class="thumb" src="'+url+'">';
    }
  });
}
setupUpload('search-upload', 'search-file', 'search-preview');
setupUpload('pro-upload', 'pro-file', 'pro-preview');

function showError(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.classList.remove('hidden');
}
function hideError(id) {
  document.getElementById(id).classList.add('hidden');
}
function setLoading(id, loading) {
  document.getElementById(id).classList.toggle('hidden', !loading);
}

function renderResults(container, results, mode) {
  const el = document.getElementById(container);
  if (!results || results.length === 0) {
    el.innerHTML = '<div style="color:#888;font-size:13px;">无结果</div>';
    return;
  }
  el.innerHTML = results.map((r, i) => {
    const score = mode === 'search' ? (r.similarity * 100).toFixed(2) + '%' : (r.score || '-');
    const scoreLabel = mode === 'search' ? '相似度' : 'VLM评分';
    return '<div class="result-card">' +
      '<img src="/img?path=' + encodeURIComponent(r.path) + '" loading="lazy">' +
      '<div class="info">' +
        '<div class="path">' + escapeHtml(r.path) + '</div>' +
        '<div><span class="rank">#' + (i+1) + '</span>' + scoreLabel + ': <span class="score">' + score + '</span></div>' +
      '</div>' +
    '</div>';
  }).join('');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderAutoBanner(id, info) {
  const el = document.getElementById(id);
  if (!info) { el.classList.add('hidden'); return; }
  const modeText = info.mode === 'auto' ? 'auto (基于库内分布)' : ('固定 ' + (info.threshold || '?'));
  const top1 = (info.top1 != null) ? Number(info.top1).toFixed(4) : '-';
  let cls = 'auto-banner';
  let icon = '✓';
  let label = '匹配充分';
  if (info.suggestPro) {
    cls += ' warn';
    icon = '💡';
    label = '建议运行 精准检索 精排';
  } else if (!info.ok) {
    cls += ' error';
    icon = '⚠';
    label = info.reason || '判定失败';
  }
  el.className = cls;
  el.innerHTML = '<div class="banner-row"><span class="banner-icon">' + icon + '</span>' +
    '<div class="banner-body">' +
      '<div class="banner-title">' + escapeHtml(label) + '</div>' +
      '<div class="banner-detail">' + escapeHtml(info.reason || '') +
        ' ｜ 阈值模式: <code>' + escapeHtml(modeText) + '</code>' +
        ' ｜ Top-1: <code>' + top1 + '</code>' +
      '</div>' +
    '</div>' +
    (info.suggestPro ? '<button class="btn" id="jump-pro">切换到精准检索 →</button>' : '') +
  '</div>';
  el.classList.remove('hidden');
  const jump = document.getElementById('jump-pro');
  if (jump) {
    jump.addEventListener('click', () => switchTab('searchpro'));
  }
}

document.getElementById('search-btn').addEventListener('click', async () => {
  const file = document.getElementById('search-file').files[0];
  if (!file) { showError('search-error', '请先上传图片'); return; }
  hideError('search-error');
  document.getElementById('search-results').innerHTML = '';
  document.getElementById('search-auto').classList.add('hidden');
  document.getElementById('search-auto').innerHTML = '';
  setLoading('search-loading', true);
  document.getElementById('search-btn').disabled = true;
  try {
    const fd = new FormData();
    fd.append('image', file);
    fd.append('topK', document.getElementById('search-topk').value);
    const resp = await fetch('/api/search', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    renderResults('search-results', data.results, 'search');
    renderAutoBanner('search-auto', data.auto);
  } catch (e) {
    showError('search-error', e.message);
  } finally {
    setLoading('search-loading', false);
    document.getElementById('search-btn').disabled = false;
  }
});

document.getElementById('pro-btn').addEventListener('click', async () => {
  const file = document.getElementById('pro-file').files[0];
  if (!file) { showError('pro-error', '请先上传图片'); return; }
  hideError('pro-error');
  document.getElementById('pro-results').innerHTML = '';
  setLoading('pro-loading', true);
  document.getElementById('pro-btn').disabled = true;
  try {
    const fd = new FormData();
    fd.append('image', file);
    fd.append('text', document.getElementById('pro-text').value);
    fd.append('topK', document.getElementById('pro-topk').value);
    const resp = await fetch('/api/search-pro', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    renderResults('pro-results', data.results, 'pro');
  } catch (e) {
    showError('pro-error', e.message);
  } finally {
    setLoading('pro-loading', false);
    document.getElementById('pro-btn').disabled = false;
  }
});

document.getElementById('build-btn').addEventListener('click', async () => {
  const dir = document.getElementById('build-dir').value.trim();
  if (!dir) { showError('build-error', '请输入目录路径'); return; }
  hideError('build-error');
  document.getElementById('build-success').classList.add('hidden');
  setLoading('build-loading', true);
  document.getElementById('build-btn').disabled = true;
  try {
    const resp = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dir, force: document.getElementById('build-force').checked })
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    const el = document.getElementById('build-success');
    el.textContent = '✅ 建库完成，当前共 ' + data.count + ' 张图片入库';
    el.classList.remove('hidden');
    loadStats();
  } catch (e) {
    showError('build-error', e.message);
  } finally {
    setLoading('build-loading', false);
    document.getElementById('build-btn').disabled = false;
  }
});

async function loadStats() {
  try {
    const resp = await fetch('/api/info');
    const data = await resp.json();
    document.getElementById('stats').textContent = '图库: ' + data.count + ' 张';
    document.getElementById('model-info').textContent = (data.model || '-') + ' · ' + data.dim + '维';
    populateTopKSelect('search-topk', data.topKDefault || 5);
    populateTopKSelect('pro-topk', data.topKDefault || 5);
  } catch (e) {}
}

// 填充返回数量下拉：auto, 1..20，默认选中 def
function populateTopKSelect(id, def) {
  const sel = document.getElementById(id);
  if (!sel) return;
  let html = '<option value="auto">auto（自适应）</option>';
  for (let i = 1; i <= 20; i++) {
    html += '<option value="' + i + '">' + i + '</option>';
  }
  sel.innerHTML = html;
  const d = parseInt(def, 10);
  sel.value = (isNaN(d) || d < 1 || d > 20) ? 'auto' : String(d);
}

async function loadInfo() {
  try {
    const resp = await fetch('/api/info');
    const data = await resp.json();
    const el = document.getElementById('info-content');
    if (data.error) {
      el.innerHTML = '<div class="error">错误: ' + escapeHtml(data.error) + '</div>';
      return;
    }
    el.innerHTML =
      '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">' +
        '<div><div style="font-size:12px;color:#888;">图片数量</div><div style="font-size:24px;font-weight:700;color:#1a73e8;">' + data.count + '</div></div>' +
        '<div><div style="font-size:12px;color:#888;">Embedding 模型</div><div style="font-size:15px;font-weight:600;margin-top:4px;">' + escapeHtml(data.model || '-') + '</div></div>' +
        '<div><div style="font-size:12px;color:#888;">向量维度</div><div style="font-size:24px;font-weight:700;color:#1a73e8;margin-top:4px;">' + data.dim + '</div></div>' +
      '</div>' +
      '<div class="desc" style="margin-top:16px;">Embedding 模型与向量维度必须保持一致，更换模型后需要重新建库。</div>';
  } catch (e) {
    document.getElementById('info-content').innerHTML = '<div class="error">加载失败: ' + escapeHtml(e.message) + '</div>';
  }
}

loadStats();
</script>
</body>
</html>`
