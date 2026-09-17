package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/joho/godotenv"
)

func main() {
	// 若 .env 不存在则自动生成模板，再加载
	ensureEnvFileSilent()
	if err := godotenv.Load(); err != nil {
		fmt.Println("⚠️  未找到 .env 文件，将尝试使用环境变量")
	}

	storeFile := os.Getenv("VECTOR_STORE_FILE")
	if storeFile == "" {
		storeFile = "./vectors.json"
	}

	if len(os.Args) < 2 {
		printUsage()
		return
	}

	cmd := os.Args[1]

	switch cmd {
	case "build":
		opts, positional := parseFlags(os.Args[2:])
		if len(positional) < 1 {
			fmt.Println("用法: agent-sight build [-w N] [-f] <图片目录>")
			return
		}
		buildLibrary(positional[0], storeFile, getFlagInt(opts, "w", 3), getFlagBool(opts, "f"))

	case "add":
		if len(os.Args) < 3 {
			fmt.Println("用法: agent-sight add <图片路径>")
			return
		}
		addImage(os.Args[2], storeFile)

	case "search":
		opts, positional := parseFlags(os.Args[2:])
		if len(positional) < 1 {
			fmt.Println("用法: agent-sight search [-k N] [-w N] [-t T] [-v] <图片...>")
			return
		}
		thresholdStr := ""
		if v, ok := opts["t"]; ok {
			thresholdStr = v
		} else {
			thresholdStr = GetAutoRerankThresholdString()
		}
		searchFast(positional, storeFile, getFlagInt(opts, "k", 5), getFlagInt(opts, "w", 10), thresholdStr, getFlagBool(opts, "v"))

	case "search-pro", "pro":
		// 精准检索：嵌入粗召回 + VLM精排（单图）
		if len(os.Args) < 3 {
			fmt.Println("用法: agent-sight search-pro <查询图片路径> [文字描述] [topK]")
			fmt.Println("示例: agent-sight search-pro ./query.jpg \"有人在笑\" 5")
			return
		}
		textQuery := ""
		topK := 0 // auto：默认不截断，全部展示
		if len(os.Args) >= 4 {
			s := strings.TrimSpace(os.Args[3])
			if strings.EqualFold(s, "auto") {
				topK = 0 // auto：不截断，全部展示
			} else if _, err := fmt.Sscanf(s, "%d", &topK); err != nil {
				textQuery = s
				if len(os.Args) >= 5 {
					s2 := strings.TrimSpace(os.Args[4])
					if strings.EqualFold(s2, "auto") {
						topK = 0
					} else {
						fmt.Sscanf(s2, "%d", &topK)
					}
				}
			}
		}
		searchPro(os.Args[2], storeFile, textQuery, topK)

	case "info":
		infoLibrary(storeFile)

	case "web":
		opts, _ := parseFlags(os.Args[2:])
		port := getFlagInt(opts, "p", 8080)
		addr := "127.0.0.1:" + strconv.Itoa(port)
		ws := NewWebServer(storeFile)
		if err := ws.Serve(addr); err != nil {
			fmt.Fprintf(os.Stderr, "❌ Web 服务启动失败: %v\n", err)
			os.Exit(1)
		}

	default:
		fmt.Printf("未知命令: %s\n\n", cmd)
		printUsage()
	}
}

// parseFlags 轻量级 flag 解析：支持 -w 3、-w=3、-f、-v 等
// 返回 (flags map, positional args)
func parseFlags(args []string) (map[string]string, []string) {
	flags := map[string]string{}
	var positional []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if strings.HasPrefix(a, "-") {
			key := strings.TrimLeft(a, "-")
			if idx := strings.Index(key, "="); idx >= 0 {
				flags[key[:idx]] = key[idx+1:]
				continue
			}
			// -k 5 形式：下一个参数就是值（如果是数字或布尔值）
			if i+1 < len(args) {
				next := args[i+1]
				if !strings.HasPrefix(next, "-") {
					flags[key] = next
					i++
					continue
				}
			}
			flags[key] = "1"
		} else {
			positional = append(positional, a)
		}
	}
	return flags, positional
}

func getFlagInt(opts map[string]string, key string, def int) int {
	if v, ok := opts[key]; ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func getFlagBool(opts map[string]string, key string) bool {
	if v, ok := opts[key]; ok {
		if v == "1" || v == "true" || v == "yes" {
			return true
		}
	}
	return false
}

func getFlagFloatOrDefault(opts map[string]string, key string, def float64) float64 {
	if v, ok := opts[key]; ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// ensureEnvFileSilent 静默版：若 .env 不存在则写入占位模板（不打印已存在提示）。
// 首次运行时会自动生成，方便用户直接跑任意命令。
func ensureEnvFileSilent() {
	if _, err := os.Stat(".env"); err == nil {
		return
	} else if !os.IsNotExist(err) {
		fmt.Println("⚠️  无法检测 .env 文件:", err)
		return
	}
	template := envTemplate()
	if err := os.WriteFile(".env", []byte(template), 0o644); err != nil {
		fmt.Println("❌ 写入 .env 失败:", err)
		return
	}
	fmt.Println("✅ 已生成 .env 模板文件（API Key 为占位符，请替换为你的真实 Key）")
	fmt.Println("   文件: ./.env")
}

// envTemplate 返回 .env 占位模板字符串，避免真实 API Key 泄露。
func envTemplate() string {
	return `# ===== OpenAI-compatible API 配置 =====
# 兼容 SiliconFlow / Moark / OpenAI 等所有 OpenAI-compatible /embeddings 端点
# ⚠️  请填写你的真实 API Key 后再运行 build / search 等命令
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen3-VL-Embedding-8B

# 常见供应商配置示例：
# SiliconFlow       : OPENAI_BASE_URL=https://api.siliconflow.cn/v1   OPENAI_MODEL=Qwen/Qwen3-VL-Embedding-8B
# Moark(模力方舟)   : OPENAI_BASE_URL=https://api.moark.com/v1        OPENAI_MODEL=Qwen/Qwen3-VL-Embedding-8B
# Paddle ModelArks  : OPENAI_MODEL=PaddlePaddle/ernie_vil-2.0-base-zh
# OpenAI            : OPENAI_BASE_URL=https://api.openai.com/v1       OPENAI_MODEL=clip-vit-large-patch14

# ===== VLM 精排模型（复用同一个 OPENAI_BASE_URL 网关）=====
VLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
# 自动精排阈值：Top-1 相似度低于此值时触发 VLM 精排
# 数字（如 0.85）= 固定阈值；'auto' = 基于库内向量分布自适应判定（推荐）
AUTO_RERANK_THRESHOLD=auto
# 粗召回数量：从向量库召回多少张候选图
RECALL_CANDIDATES=50
# 单次送 VLM 精排的候选数：从召回结果里挑最相似的 N 张发给 VLM（避免请求体过大）
VLM_BATCH_SIZE=6
# VLM 请求超时（秒）：超时后自动降级返回嵌入结果
VLM_TIMEOUT=120

# ===== 向量库存储文件（本地 JSON 持久化）=====
VECTOR_STORE_FILE=./vectors.json

# ===== 图片预处理模式 =====
# compress（默认）：大图自动缩放并转 JPEG 压缩到 IMAGE_MAX_MB MB 以内
# raw             ：不做任何处理，原样 base64 发送
IMAGE_OPTIMIZE=compress
# 最大边长（像素），仅 compress 模式生效
IMAGE_MAX_SIDE=1024
# 压缩后目标字节上限（MB），仅 compress 模式生效
IMAGE_MAX_MB=2
`
}

func printUsage() {
	fmt.Println("🖼️  多模态RAG两级检索 以图搜图工具")
	fmt.Println("")
	fmt.Println("📚 建库命令:")
	fmt.Println("  agent-sight build [-w N] [-f] <图片目录>")
	fmt.Println("      -w N  并发数，默认 3")
	fmt.Println("      -f    强制重建（忽略已存在记录，默认增量构建）")
	fmt.Println("")
	fmt.Println("  agent-sight add <图片路径>    追加单张图片到库")
	fmt.Println("  agent-sight info              查看向量库信息")
	fmt.Println("")
	fmt.Println("🔍 检索命令:")
	fmt.Println("  agent-sight search [-k N] [-w N] [-t T] [-v] <图片...>")
	fmt.Println("      -k N  返回数量，默认 5")
	fmt.Println("      -w N  并发数，默认 10（多图并发向量化）")
	fmt.Println("      -t T  相似度阈值，低于此值提示运行 search-pro 精排。")
	fmt.Println("           数字(如 0.85) = 固定阈值；'auto' = 基于库内向量分布自适应判定")
	fmt.Println("      -v    详细模式：输出 相似度 排名 路径")
	fmt.Println("      支持多图输入，默认按相似度降序输出路径")
	fmt.Println("")
	fmt.Println("  agent-sight search-pro <图片> [文字描述] [topK]")
	fmt.Println("      精准检索：嵌入粗召回 + VLM大模型精排/二次查找")
	fmt.Println("")
	fmt.Println("🌐 Web 服务器命令:")
	fmt.Println("  agent-sight web [-p PORT]")
	fmt.Println("      -p PORT 监听端口，默认 8080，启动后自动打开默认浏览器")
	fmt.Println("")
	fmt.Println("💡 示例:")
	fmt.Println("  agent-sight build ./input")
	fmt.Println("  agent-sight build -w 5 ./input")
	fmt.Println("  agent-sight search ./test.jpg")
	fmt.Println("  agent-sight search -k 10 -v ./test.jpg")
	fmt.Println("  agent-sight search ./test/1.jpg ./test/2.jpg")
}

// ===== 建库 =====
func buildLibrary(dir string, storeFile string, workers int, force bool) {
	provider, err := NewEmbedProvider()
	if err != nil {
		fmt.Println("❌ 初始化嵌入服务失败:", err)
		os.Exit(1)
	}

	vs, err := LoadVectorStore(storeFile)
	if err != nil {
		fmt.Println("❌ 加载向量库失败:", err)
		os.Exit(1)
	}
	if err := vs.CheckModel(provider.ModelName(), provider.Dim()); err != nil {
		fmt.Println("❌", err)
		os.Exit(1)
	}

	// 收集已入库路径，用于增量构建
	existing := map[string]bool{}
	if force {
		// 强制重建：清空旧记录，避免同路径重复累加
		fmt.Printf("🔄 强制重建模式，清空原有 %d 条记录\n", len(vs.Records))
		vs.Records = nil
	} else {
		for _, r := range vs.Records {
			// 归一化路径便于比较
			existing[filepath.ToSlash(r.Path)] = true
		}
	}

	var imageFiles []string
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp" || ext == ".bmp" {
			imageFiles = append(imageFiles, path)
		}
		return nil
	})

	if len(imageFiles) == 0 {
		fmt.Println("⚠️  目录下没有找到图片")
		return
	}

	// 过滤已存在
	var toProcess []string
	skip := 0
	for _, p := range imageFiles {
		if !force && existing[filepath.ToSlash(p)] {
			skip++
			continue
		}
		toProcess = append(toProcess, p)
	}

	fmt.Printf("🚀 开始构建向量库，目录共 %d 张，已存在 %d 张，本次处理 %d 张\n",
		len(imageFiles), skip, len(toProcess))
	fmt.Printf("   模型: %s  并发: %d\n\n", provider.ModelName(), workers)

	if len(toProcess) == 0 {
		fmt.Println("✅ 所有图片均已入库，无需处理。加 -f 强制重建。")
		return
	}

	// 并发处理
	var (
		mu       sync.Mutex
		wg       sync.WaitGroup
		success  int
		failed   int
		stopOnce sync.Once
		fatalErr error
	)
	total := len(toProcess)
	jobCh := make(chan string, workers)

	// worker 池
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range jobCh {
				vec, err := provider.EmbedImage(path)
				mu.Lock()
				if err != nil {
					failed++
					cur := failed
					mu.Unlock()
					fmt.Printf("[FAIL %d/%d] %s: %v\n", cur, total, path, err)
					if isFatalAPIError(err) {
						stopOnce.Do(func() {
							fatalErr = err
							close(jobCh)
						})
						return
					}
					continue
				}
				vs.Add(path, vec)
				success++
				cur := success
				mu.Unlock()
				fmt.Printf("[OK %d/%d] %s (%d维)\n", cur, total, path, len(vec))
			}
		}()
	}

	// 分发任务：生产者，遇到致命错误就关闭 channel 让 worker 自然退出
	for _, p := range toProcess {
		mu.Lock()
		if fatalErr != nil {
			mu.Unlock()
			break
		}
		mu.Unlock()
		jobCh <- p
	}
	close(jobCh)
	wg.Wait()

	if fatalErr != nil {
		fmt.Fprintf(os.Stderr, "\n❌ API 请求失败，终止建库。请检查 .env 中的 API Key / BASE_URL / MODEL 配置。\n   %v\n", fatalErr)
		os.Exit(1)
	}

	if err := vs.Save(storeFile); err != nil {
		fmt.Println("\n❌ 保存向量库失败:", err)
		os.Exit(1)
	}

	fmt.Printf("\n✅ 完成！本次成功 %d 张，失败 %d 张，跳过 %d 张，库中总共 %d 张\n",
		success, failed, skip, vs.Count())
	fmt.Printf("   向量库文件: %s\n", storeFile)
}

func addImage(path string, storeFile string) {
	provider, err := NewEmbedProvider()
	if err != nil {
		fmt.Println("❌ 初始化嵌入服务失败:", err)
		os.Exit(1)
	}
	vs, err := LoadVectorStore(storeFile)
	if err != nil {
		fmt.Println("❌ 加载向量库失败:", err)
		os.Exit(1)
	}
	if err := vs.CheckModel(provider.ModelName(), provider.Dim()); err != nil {
		fmt.Println("❌", err)
		os.Exit(1)
	}
	fmt.Printf("正在向量化: %s ... ", path)
	vec, err := provider.EmbedImage(path)
	if err != nil {
		fmt.Println("失败:", err)
		os.Exit(1)
	}
	vs.Add(path, vec)
	if err := vs.Save(storeFile); err != nil {
		fmt.Println("保存失败:", err)
		os.Exit(1)
	}
	fmt.Printf("✓ 已入库 (%d维)\n", len(vec))
}

func infoLibrary(storeFile string) {
	vs, err := LoadVectorStore(storeFile)
	if err != nil {
		fmt.Println("❌ 加载向量库失败:", err)
		os.Exit(1)
	}
	fmt.Println("📊 向量库信息")
	fmt.Println("   文件:", storeFile)
	fmt.Println("   嵌入模型:", vs.ModelName)
	fmt.Println("   向量维度:", vs.Dim)
	fmt.Println("   VLM精排模型:", os.Getenv("VLM_MODEL"))
	fmt.Println("   自动精排阈值:", GetAutoRerankThresholdString())
	fmt.Println("   粗召回数量:", GetRecallCandidates())
	fmt.Println("   图片总数:", vs.Count())
}

// ===== 快速检索（仅嵌入，支持多图并发向量化）=====
func searchFast(queryPaths []string, storeFile string, topK int, workers int, thresholdStr string, verbose bool) {
	provider, err := NewEmbedProvider()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 初始化嵌入服务失败: %v\n", err)
		os.Exit(1)
	}
	vs, err := LoadVectorStore(storeFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 加载向量库失败: %v\n", err)
		os.Exit(1)
	}
	if vs.Count() == 0 {
		fmt.Fprintf(os.Stderr, "⚠️  向量库为空，请先运行 build 命令建库\n")
		os.Exit(1)
	}

	// 阈值模式：auto = 基于库内向量分布自适应，否则解析为固定值
	autoMode := strings.EqualFold(strings.TrimSpace(thresholdStr), "auto")
	var threshold float64
	if !autoMode {
		f, err := strconv.ParseFloat(strings.TrimSpace(thresholdStr), 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ -t 参数必须是数字或 'auto'，收到: %q\n", thresholdStr)
			os.Exit(1)
		}
		threshold = f
	}

	type queryResult struct {
		Path    string
		Results []SearchResult
	}
	resultsCh := make(chan queryResult, len(queryPaths))
	var wg sync.WaitGroup

	// worker 池限制并发向量化数，默认 10
	jobCh := make(chan string, len(queryPaths))
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for qp := range jobCh {
				vec, err := provider.EmbedImage(qp)
				if err != nil {
					fmt.Fprintf(os.Stderr, "❌ %s 向量化失败: %v\n", qp, err)
					resultsCh <- queryResult{Path: qp}
					continue
				}
				resultsCh <- queryResult{Path: qp, Results: vs.Search(vec, topK)}
			}
		}()
	}
	for _, qp := range queryPaths {
		jobCh <- qp
	}
	close(jobCh)
	wg.Wait()
	close(resultsCh)

	for qr := range resultsCh {
		if len(qr.Results) == 0 {
			continue
		}
		if verbose {
			// 详细模式：查询图 | 相似度(高→低) 排名 路径
			for i, r := range qr.Results {
				fmt.Printf("%s\t%.6f\t%d\t%s\n", qr.Path, r.Similarity, i+1, r.Path)
			}
		} else {
			for _, r := range qr.Results {
				fmt.Println(r.Path)
			}
		}

		top1 := float64(qr.Results[0].Similarity)
		needPro := false
		reason := ""

		if autoMode {
			// auto 模式：用库内向量两两相似度分布的 p90 作为动态阈值
			p90, distN := computeAutoThreshold(vs)
			if distN == 0 {
				needPro = false
				reason = "库太小，跳过 auto 阈值判定"
			} else {
				gap := top1 - float64(qr.Results[1].Similarity)
				if top1 < p90 || (gap < 0.05 && top1 < p90+0.05) {
					needPro = true
					reason = fmt.Sprintf("库内分布 p90=%.4f, Top-1=%.4f (差 %.4f)", p90, top1, gap)
				} else {
					reason = fmt.Sprintf("库内分布 p90=%.4f, Top-1=%.4f 匹配充分", p90, top1)
				}
			}
			if verbose {
				fmt.Fprintf(os.Stderr, "ℹ️  [%s] auto 阈值判定: %s\n", qr.Path, reason)
			}
		} else {
			if top1 < threshold {
				needPro = true
			}
			if verbose {
				fmt.Fprintf(os.Stderr, "ℹ️  [%s] 固定阈值判定: Top-1=%.4f, 阈值=%.4f\n", qr.Path, top1, threshold)
			}
		}

		if needPro {
			fmt.Fprintf(os.Stderr, "💡 [%s] 建议运行 search-pro 精排: %s\n", qr.Path, reason)
		}
	}
}

// computeAutoThreshold 基于库内向量两两余弦相似度分布，返回 p90 分位值。
// 库超过 2000 张时随机抽样到 2000 张以控制计算量。
func computeAutoThreshold(vs *VectorStore) (float64, int) {
	n := vs.Count()
	if n < 2 {
		return 0, 0
	}
	// 随机抽样
	sample := vs.Records
	if n > 2000 {
		// Fisher-Yates 前 2000
		sample = append([]ImageRecord{}, vs.Records...)
		for i := n - 1; i > 1999; i-- {
			j := (i * 2654435761) % (i + 1) // 伪随机索引
			sample[i], sample[j] = sample[j], sample[i]
		}
		sample = sample[:2000]
	}
	sims := make([]float64, 0, len(sample)*(len(sample)-1)/2)
	for i := 0; i < len(sample); i++ {
		for j := i + 1; j < len(sample); j++ {
			sims = append(sims, float64(cosineSimilarity(sample[i].Vector, sample[j].Vector)))
		}
	}
	if len(sims) == 0 {
		return 0, 0
	}
	sort.Float64s(sims)
	idx := int(float64(len(sims)) * 0.9)
	if idx >= len(sims) {
		idx = len(sims) - 1
	}
	return sims[idx], len(sims)
}

// ===== 精准检索（嵌入粗召回 + VLM精排）=====
func searchPro(queryPath string, storeFile string, textQuery string, topK int) {
	provider, err := NewEmbedProvider()
	if err != nil {
		fmt.Println("❌ 初始化嵌入服务失败:", err)
		os.Exit(1)
	}
	vs, err := LoadVectorStore(storeFile)
	if err != nil {
		fmt.Println("❌ 加载向量库失败:", err)
		os.Exit(1)
	}
	if vs.Count() == 0 {
		fmt.Println("⚠️  向量库为空，请先运行 build 命令建库")
		os.Exit(1)
	}

	reranker := NewReranker()
	recallN := GetRecallCandidates()
	if recallN > vs.Count() {
		recallN = vs.Count()
	}
	batchSize := GetVLMBatchSize()
	if batchSize > recallN {
		batchSize = recallN
	}

	fmt.Printf("🎯 精准检索: %s\n", queryPath)
	if textQuery != "" {
		fmt.Printf("   筛选条件: %s\n", textQuery)
	}
	fmt.Printf("   图库大小: %d 张\n", vs.Count())
	if recallN >= vs.Count() {
		fmt.Printf("   流程: 嵌入召回全部 %d 张 → VLM精排Top-auto（送 %d 张打分）\n", vs.Count(), batchSize)
	} else {
		fmt.Printf("   流程: 嵌入召回Top-%d → VLM精排Top-auto（送 %d 张打分）\n", recallN, batchSize)
	}

	thresholdStr := GetAutoRerankThresholdString()
	if strings.EqualFold(thresholdStr, "auto") {
		p90, distN := computeAutoThreshold(vs)
		if distN == 0 {
			fmt.Printf("   阈值: auto（库太小，跳过 p90 判定）\n")
		} else {
			fmt.Printf("   阈值: auto（库内分布 p90=%.4f，样本 %d 对）\n", p90, distN)
		}
	} else {
		fmt.Printf("   阈值: %s\n", thresholdStr)
	}
	fmt.Println()

	// 第一步：嵌入粗召回
	fmt.Println("① 嵌入粗召回中...")
	vec, err := provider.EmbedImage(queryPath)
	if err != nil {
		fmt.Println("❌ 查询图片向量化失败:", err)
		os.Exit(1)
	}
	candidates := vs.Search(vec, recallN)
	if len(candidates) > 0 {
		top1 := float64(candidates[0].Similarity)
		sent := len(candidates)
		if batchSize > 0 && batchSize < sent {
			sent = batchSize
		}
		if recallN >= vs.Count() {
			fmt.Printf("   已召回全部 %d 张（Top-1 相似度=%.4f），送 VLM 精排前 %d 张\n", len(candidates), top1, sent)
		} else {
			fmt.Printf("   已召回 %d/%d 张（Top-1 相似度=%.4f），送 VLM 精排前 %d 张\n", len(candidates), recallN, top1, sent)
		}
	} else {
		fmt.Printf("   已召回 0 张候选图\n")
	}
	fmt.Println()

	// 第二步：VLM精排
	fmt.Println("② VLM大模型精排中（请稍候2-5秒）...")
	rerankResults, err := reranker.Rerank(queryPath, candidates, textQuery)
	if err != nil {
		fmt.Println("❌ VLM精排失败:", err)
		fmt.Println("💡 降级返回嵌入检索结果:")
		downLimit := topK
		if downLimit <= 0 {
			downLimit = len(candidates)
		}
		for i, r := range candidates {
			if i >= downLimit {
				break
			}
			fmt.Printf("  %d. [%.4f] %s\n", i+1, r.Similarity, r.Path)
		}
		os.Exit(1)
	}
	fmt.Println("   ✓ 精排完成\n\n")

	// 输出最终结果
	topKLabel := topKToString(topK)
	fmt.Printf("🏆 VLM精排 Top-%s 结果（0-10分制）：\n\n", topKLabel)
	for i, r := range rerankResults {
		if topK > 0 && i >= topK {
			break
		}
		scoreBar := strings.Repeat("█", int(r.Score/10*30)) + strings.Repeat("░", 30-int(r.Score/10*30))
		matchMark := "✓"
		if !r.IsMatch {
			matchMark = "✗"
		}
		fmt.Printf("  %d. [%s%.1f分] %s\n", i+1, matchMark, r.Score, scoreBar)
		fmt.Printf("      📄 %s\n", r.Path)
		fmt.Printf("      💬 %s\n\n", r.Reason)
	}
}

// topKToString 把 topK 数值转成显示标签：0 → auto，其它 → 数字
func topKToString(topK int) string {
	if topK <= 0 {
		return "auto"
	}
	return strconv.Itoa(topK)
}
