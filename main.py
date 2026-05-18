import os
import sys
import threading
import time
from pathlib import Path
import urllib.request
import json
from urllib.parse import urlparse, urljoin
import asyncio
import logging


import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import networkx as nx

# 引入 Scrapling 核心组件
try:
    from scrapling.fetchers import Fetcher, DynamicFetcher

    from scrapling.core.shell import Convertor
except ImportError:
    messagebox.showerror("依赖缺失", "未检测到 scrapling，请确保环境正确。")
    sys.exit(1)

# RAG 依赖
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    messagebox.showerror("依赖缺失", "未检测到 chromadb 或 sentence-transformers。")
    sys.exit(1)


# 配置默认保存路径
DEFAULT_OUTPUT_DIR = Path.home() / "Scrapling_Outputs"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 确保 ChromaDB 目录存在
CHROMA_DB_DIR = DEFAULT_OUTPUT_DIR / "chroma_db"
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


class TextRedirector:
    """用于重定向标准输出(stdout)到 CustomTkinter Text 文本框的辅助类"""
    def __init__(self, text_widget, update_callback=None):
        self.text_widget = text_widget
        self.update_callback = update_callback

    def write(self, str_val):
        if str_val.strip() or str_val == '\n':
            if self.update_callback:
                self.update_callback(self.text_widget, str_val)

    def flush(self):
        pass


class FullSiteCrawler:
    """
    全站递归爬虫，基于内置的 Queue 和 Scrapling 原生 Fetcher/DynamicFetcher
    """
    def __init__(self, start_url, output_dir, engine, css_selector, format_ext, update_graph_cb, log_cb, max_depth=5):
        self.start_url = start_url
        self.allowed_domain = urlparse(start_url).netloc
        self.output_dir = output_dir
        self.engine = engine
        self.css_selector = css_selector if css_selector else None
        self.format_ext = format_ext
        self.update_graph_cb = update_graph_cb
        self.log_cb = log_cb
        self.max_depth = max_depth
        self.visited = set()

        import queue
        self.queue = queue.Queue()

    def start(self):
        # 初始入队
        self.queue.put((self.start_url, 0, None))
        self.visited.add(self.start_url)

        while not self.queue.empty():
            current_url, depth, source_url = self.queue.get()
            self._process_url(current_url, depth, source_url)

    def _process_url(self, url, depth, source_url):
        self.log_cb(f"已爬取: {url}")

        try:
            response = None
            if self.engine == "Static":
                response = Fetcher.get(url)
            else:
                response = DynamicFetcher.fetch(url, real_chrome=False)

            if not response or (hasattr(response, 'status') and response.status >= 400):
                self.log_cb(f"⚠️ 跳过: {url} (页面响应失败)")
                return

            # Extract title
            title = ""
            if hasattr(response, 'css'):
                title_tag = response.css('title::text')
                if title_tag:
                    title = title_tag[0] if type(title_tag) is list else title_tag

            if not title:
                title = url.split('/')[-1]

            title = title.strip() if isinstance(title, str) else str(title).strip()


            # Use path part of URL for unique filename if possible
            path_part = urlparse(url).path.strip('/').replace('/', '_')
            filename = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            if not filename:
                filename = f"page_{len(self.visited)}"
            filename = filename.replace(" ", "_")[:30]
            if path_part:
                filename = f"{filename}_{path_part}"[:50]

            full_path = self.output_dir / f"{filename}{self.format_ext}"

            # 使用 Convertor 提取并保存内容
            Convertor.write_content_to_file(response, str(full_path), self.css_selector)

            self.log_cb(f"✅ 保存文件: {filename}{self.format_ext}")

            # 触发 RAG 索引更新和图谱更新
            self.update_graph_cb(source_url, response.url, title, str(full_path))

        except Exception as e:
            self.log_cb(f"❌ 保存失败 {response.url}: {str(e)}")

        # 检查是否达到最大深度
        if depth >= self.max_depth:
            return

        # 提取当前页面的所有链接
        for a_tag in response.css('a'):
            href = a_tag.attrib.get('href')
            if href:
                full_url = urljoin(response.url, href)
                full_url = full_url.split('#')[0] # Remove fragment

                parsed_url = urlparse(full_url)
                if parsed_url.netloc == self.allowed_domain and full_url not in self.visited:
                    self.visited.add(full_url)
                    self.queue.put((full_url, depth + 1, response.url))


class ScraplingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Scrapling 智能爬虫分析中心")
        self.geometry("1400x800")
        self.minsize(1000, 700)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # 初始化图谱数据
        self.graph = nx.DiGraph()
        self.node_positions = {}

        # 初始化队列用于线程间通信
        import queue
        self.ui_queue = queue.Queue()

        # RAG 组件
        self.chroma_client = None
        self.collection = None
        self.embed_model = None
        self.rag_ready = False

        self._create_widgets()

        # 拦截标准输出
        sys.stdout = TextRedirector(self.log_text, self._safe_update_text)
        sys.stderr = TextRedirector(self.log_text, self._safe_update_text)

        self.refresh_file_list()

        # 启动后台初始化 RAG
        threading.Thread(target=self._init_rag_system, daemon=True).start()

        # 定时器处理 UI 队列
        self.after(100, self._process_ui_queue)

        print(f"系统初始化完毕。文档默认存储目录：{DEFAULT_OUTPUT_DIR}\n")

    def _init_rag_system(self):
        self.ui_queue.put(("log", "🔄 正在初始化 AI RAG 向量数据库和本地嵌入模型...\n"))
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            self.collection = self.chroma_client.get_or_create_collection(name="scrapling_docs")

            # 使用轻量级多语言支持的模型
            self.embed_model = SentenceTransformer('shibing624/text2vec-base-chinese')

            self.rag_ready = True
            self.ui_queue.put(("log", "✅ AI RAG 系统初始化成功！\n"))
            self.ui_queue.put(("rag_status", "正常"))

            # 将已有文件加载到数据库中
            self._index_existing_files()

        except Exception as e:
            self.ui_queue.put(("log", f"❌ RAG 初始化失败: {str(e)}\n"))
            self.ui_queue.put(("rag_status", "错误"))

    def _index_existing_files(self):
        if not self.rag_ready: return
        self.ui_queue.put(("log", "🔄 正在扫描并索引本地已有文档...\n"))

        valid_extensions = {".md", ".html", ".txt"}
        count = 0
        for p in DEFAULT_OUTPUT_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in valid_extensions:
                self._add_file_to_index(str(p))
                count += 1
        self.ui_queue.put(("log", f"✅ 本地文档索引完成，共处理 {count} 个文件。\n"))

    def _add_file_to_index(self, filepath):
        if not self.rag_ready: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 简单的文本切片
            chunk_size = 500
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

            if not chunks: return

            embeddings = self.embed_model.encode(chunks).tolist()
            ids = [f"{Path(filepath).name}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filepath} for _ in range(len(chunks))]

            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Index error for {filepath}: {e}")

    def _safe_update_text(self, text_widget, text):
        self.ui_queue.put(("text", (text_widget, text)))

    def _process_ui_queue(self):
        while not self.ui_queue.empty():
            msg_type, data = self.ui_queue.get()
            if msg_type == "text":
                widget, text = data
                widget.configure(state='normal')
                widget.insert(tk.END, text)
                widget.see(tk.END)
                widget.configure(state='disabled')
            elif msg_type == "log":
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, data)
                self.log_text.see(tk.END)
                self.log_text.configure(state='disabled')
            elif msg_type == "rag_status":
                self.rag_status_lbl.configure(text=f"RAG引擎状态: {data}")
            elif msg_type == "graph_update":
                self._draw_graph()
            elif msg_type == "add_node":
                self._safe_add_node(*data)
            elif msg_type == "crawl_done":
                self.start_btn.configure(state="normal", text="🚀 开始全站爬取")
                self.refresh_file_list()

        self.after(100, self._process_ui_queue)

    def _create_widgets(self):
        # 整体布局：分为左、中、右三个 Frame
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左侧控制面板 =================
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(left_frame, text="爬取参数配置", font=("Microsoft YaHei", 16, "bold")).pack(pady=10)

        # URL
        ctk.CTkLabel(left_frame, text="目标网址 (URL):").pack(anchor="w", padx=10)
        self.url_entry = ctk.CTkEntry(left_frame, placeholder_text="https://quotes.toscrape.com")
        self.url_entry.insert(0, "https://quotes.toscrape.com")
        self.url_entry.pack(fill="x", padx=10, pady=5)

        # Max Depth
        ctk.CTkLabel(left_frame, text="最大爬取深度:").pack(anchor="w", padx=10)
        self.depth_var = ctk.StringVar(value="3")
        self.depth_cb = ctk.CTkComboBox(left_frame, variable=self.depth_var, values=["1", "3", "5", "10"])
        self.depth_cb.pack(fill="x", padx=10, pady=5)

        # Engine
        ctk.CTkLabel(left_frame, text="抓取引擎:").pack(anchor="w", padx=10, pady=(10,0))
        self.engine_var = ctk.StringVar(value="Static")
        ctk.CTkRadioButton(left_frame, text="静态高速度", variable=self.engine_var, value="Static").pack(anchor="w", padx=20, pady=2)
        ctk.CTkRadioButton(left_frame, text="动态渲染 (Chrome)", variable=self.engine_var, value="Dynamic").pack(anchor="w", padx=20, pady=2)

        # Format
        ctk.CTkLabel(left_frame, text="保存格式:").pack(anchor="w", padx=10, pady=(10,0))
        self.format_var = ctk.StringVar(value=".md")
        self.format_cb = ctk.CTkComboBox(left_frame, variable=self.format_var, values=[".md", ".html", ".txt"])
        self.format_cb.pack(fill="x", padx=10, pady=5)

        # CSS
        ctk.CTkLabel(left_frame, text="CSS筛选器 (可选):").pack(anchor="w", padx=10, pady=(10,0))
        self.css_entry = ctk.CTkEntry(left_frame, placeholder_text="如: .quote，留空抓整页")
        self.css_entry.pack(fill="x", padx=10, pady=5)

        # Start Button
        self.start_btn = ctk.CTkButton(left_frame, text="🚀 开始全站爬取", height=40, font=("Microsoft YaHei", 14, "bold"), command=self.start_crawl_thread)
        self.start_btn.pack(fill="x", padx=10, pady=20)


        # ================= 中间区域 (Tabview) =================
        middle_frame = ctk.CTkTabview(self)
        middle_frame.grid(row=0, column=1, padx=(0,10), pady=10, sticky="nsew")

        middle_frame.add("日志输出")
        middle_frame.add("双链图谱")
        middle_frame.add("AI RAG 查询")

        # --- 日志输出 Tab ---
        log_tab = middle_frame.tab("日志输出")
        log_tab.grid_columnconfigure(0, weight=1)
        log_tab.grid_rowconfigure(0, weight=1)
        self.log_text = ctk.CTkTextbox(log_tab, font=("Consolas", 12))
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.log_text.configure(state='disabled')

        # --- 双链图谱 Tab ---
        graph_tab = middle_frame.tab("双链图谱")
        graph_tab.grid_columnconfigure(0, weight=1)
        graph_tab.grid_rowconfigure(0, weight=1)

        self.graph_canvas = tk.Canvas(graph_tab, bg="#1e1e1e", highlightthickness=0)
        self.graph_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 绑定重绘事件处理大小变化
        self.graph_canvas.bind("<Configure>", lambda e: self.ui_queue.put(("graph_update", None)))

        # --- AI RAG 查询 Tab ---
        rag_tab = middle_frame.tab("AI RAG 查询")
        rag_tab.grid_columnconfigure(0, weight=1)
        rag_tab.grid_rowconfigure(1, weight=1)

        top_rag_frame = ctk.CTkFrame(rag_tab, fg_color="transparent")
        top_rag_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(top_rag_frame, text="DeepSeek API Key:").pack(side="left")
        self.api_key_entry = ctk.CTkEntry(top_rag_frame, show="*", width=300, placeholder_text="sk-...")
        self.api_key_entry.pack(side="left", padx=10)

        self.rag_status_lbl = ctk.CTkLabel(top_rag_frame, text="RAG引擎状态: 初始化中...")
        self.rag_status_lbl.pack(side="right", padx=10)

        self.chat_display = ctk.CTkTextbox(rag_tab, font=("Microsoft YaHei", 12))
        self.chat_display.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.chat_display.configure(state='disabled')

        bottom_rag_frame = ctk.CTkFrame(rag_tab, fg_color="transparent")
        bottom_rag_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.chat_input = ctk.CTkEntry(bottom_rag_frame, placeholder_text="向 AI 提问关于已爬取内容的问题...")
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.chat_input.bind("<Return>", lambda e: self.send_ai_query())

        ctk.CTkButton(bottom_rag_frame, text="发送", command=self.send_ai_query).pack(side="right")


        # ================= 右侧文档管理器 =================
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=2, padx=(0,10), pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right_frame, text="已保存文档", font=("Microsoft YaHei", 16, "bold")).grid(row=0, column=0, pady=10)

        # Treeview (CTk 中使用 tk.Listbox 或原生 Treeview 比较好，这里用原生 ttk.Treeview 结合 CTk 样式)
        from tkinter import ttk
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.map('Treeview', background=[('selected', '#1f538d')])

        self.file_tree = ttk.Treeview(right_frame, columns=("name", "size"), show="headings")
        self.file_tree.heading("name", text="文件名")
        self.file_tree.heading("size", text="大小")
        self.file_tree.column("name", width=150)
        self.file_tree.column("size", width=50, anchor="e")
        self.file_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.file_tree.bind("<Double-1>", lambda e: self.open_selected_file())

        btn_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkButton(btn_frame, text="打开目录", command=self.open_output_dir).pack(side="left", fill="x", expand=True, padx=(0,5))
        ctk.CTkButton(btn_frame, text="删除文档", command=self.delete_selected_file, fg_color="#d32f2f", hover_color="#b71c1c").pack(side="right", fill="x", expand=True, padx=(5,0))

    # ================= 业务逻辑 =================

    def start_crawl_thread(self):
        url = self.url_entry.get().strip()
        if not url.startswith(("http://", "https://")):
            messagebox.showwarning("格式有误", "请输入有效网址")
            return

        self.start_btn.configure(state="disabled", text="⏳ 爬取进行中...")
        self.graph.clear()
        self.node_positions.clear()
        self.ui_queue.put(("graph_update", None))

        # 获取所有 Tkinter 变量，避免在子线程中调用 .get() 导致死锁
        engine = self.engine_var.get()
        fmt = self.format_var.get()
        css = self.css_entry.get().strip()
        depth = int(self.depth_var.get())

        # 启动线程
        threading.Thread(target=self.crawl_process, args=(url, engine, fmt, css, depth), daemon=True).start()

    def crawl_process(self, url, engine, fmt, css, depth):
        print(f"\n{'='*40}")
        print(f"🚀 新全站任务启动: {url}")
        print(f"⚙️ 引擎: {engine}, 深度: {depth}, 格式: {fmt}")
        print("="*40)

        try:
            # 记录起始节点
            self.graph.add_node(url, title="Start", path="")

            spider = FullSiteCrawler(
                start_url=url,
                output_dir=DEFAULT_OUTPUT_DIR,
                engine=engine,
                css_selector=css,
                format_ext=fmt,
                update_graph_cb=self._on_node_scraped,
                log_cb=lambda msg: self.ui_queue.put(("log", msg + "\n")),
                max_depth=depth
            )

            spider.start()

            print("\n🎉 全站爬取任务完成！")
        except Exception as e:
            print(f"\n❌ 爬取发生错误: {e}")
        finally:
            self.ui_queue.put(("crawl_done", None))

    def _on_node_scraped(self, source_url, current_url, title, file_path):
        """爬虫每爬取一个节点后的回调，用于更新图谱和 RAG 索引"""
        # 我们把图谱更新任务也塞进队列，交由主线程安全执行，避免 RuntimeError (dict changed size during iteration)
        self.ui_queue.put(("add_node", (source_url, current_url, title, file_path)))

        # 增量索引到 RAG
        if self.rag_ready:
            self._add_file_to_index(file_path)

    def _safe_add_node(self, source_url, current_url, title, file_path):
        """在主线程执行图谱增量更新"""
        if current_url not in self.graph:
            self.graph.add_node(current_url, title=title, path=file_path)

        if source_url and source_url != current_url:
            self.graph.add_edge(source_url, current_url)

        # 设置防抖：如果在短时间内连续爬取了多个页面，不必每次都重绘
        if hasattr(self, "_graph_update_after_id"):
            self.after_cancel(self._graph_update_after_id)
        self._graph_update_after_id = self.after(300, self._draw_graph)

    def _draw_graph(self):
        """将 NetworkX 图绘制到 Canvas 上"""
        self.graph_canvas.delete("all")
        if len(self.graph.nodes) == 0:
            return

        w = self.graph_canvas.winfo_width()
        h = self.graph_canvas.winfo_height()
        if w < 10 or h < 10: return

        # 为了避免阻塞主线程太久，当节点超过一定数量时，减少迭代次数或者固定布局
        iterations = 20 if len(self.graph.nodes) < 50 else 5
        # 使用 spring 布局计算位置
        self.node_positions = nx.spring_layout(self.graph, pos=self.node_positions, iterations=iterations, k=0.5)

        xs = [p[0] for p in self.node_positions.values()]
        ys = [p[1] for p in self.node_positions.values()]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1

        padding = 30

        def scale(p):
            x = padding + (p[0] - min_x) / range_x * (w - 2*padding)
            y = padding + (p[1] - min_y) / range_y * (h - 2*padding)
            return x, y

        # 绘制边
        for u, v in self.graph.edges:
            if u in self.node_positions and v in self.node_positions:
                x1, y1 = scale(self.node_positions[u])
                x2, y2 = scale(self.node_positions[v])
                self.graph_canvas.create_line(x1, y1, x2, y2, fill="#555555", width=1.5, arrow=tk.LAST)

        # 绘制节点
        for n in self.graph.nodes:
            if n in self.node_positions:
                x, y = scale(self.node_positions[n])
                # 节点为蓝色圆圈
                r = 6
                self.graph_canvas.create_oval(x-r, y-r, x+r, y+r, fill="#1a73e8", outline="#ffffff", width=2)
                # 节点文字
                title = self.graph.nodes[n].get("title", "")
                if title:
                    # 缩略标题
                    short_title = title[:10] + "..." if len(title) > 10 else title
                    self.graph_canvas.create_text(x, y+15, text=short_title, fill="#aaaaaa", font=("Microsoft YaHei", 8))

    def send_ai_query(self):
        query = self.chat_input.get().strip()
        api_key = self.api_key_entry.get().strip()

        if not query: return
        if not api_key:
            messagebox.showwarning("提示", "请输入 DeepSeek API Key")
            return
        if not self.rag_ready:
            messagebox.showwarning("提示", "AI 向量库仍在初始化中，请稍后。")
            return

        self.chat_input.delete(0, tk.END)
        self._append_chat(f"你: {query}\n")

        threading.Thread(target=self._process_ai_query, args=(api_key, query), daemon=True).start()

    def _process_ai_query(self, api_key, query):
        try:
            self._append_chat("AI: [正在检索本地知识库...]\n")

            # 1. 向量检索
            query_embedding = self.embed_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=3
            )

            context = ""
            if results and results['documents'] and results['documents'][0]:
                for doc in results['documents'][0]:
                    context += doc + "\n...\n"

            if not context.strip():
                context = "没有找到相关的本地爬取数据。"

            # 2. 构造 Prompt
            prompt = f"基于以下本地爬取的文档片段回答我的问题。如果文档中没有相关信息，请说明。\n\n【本地文档片段】：\n{context}\n\n【我的问题】：{query}"

            # 3. 请求 DeepSeek API
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个基于用户本地知识库解答问题的AI助手。"},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }

            req = urllib.request.Request(url, json.dumps(data).encode('utf-8'), headers)

            self._append_chat("AI: [正在思考...]\n")

            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                answer = result['choices'][0]['message']['content']

                self._append_chat(f"AI: {answer}\n\n")

        except urllib.error.HTTPError as e:
            if e.code == 401:
                self._append_chat("❌ API Key 错误或未授权 (HTTP 401)\n\n")
            elif e.code == 402:
                 self._append_chat("❌ API 账户余额不足 (HTTP 402)\n\n")
            else:
                self._append_chat(f"❌ 请求 API 失败: {e}\n\n")
        except Exception as e:
            self._append_chat(f"❌ 发生错误: {str(e)}\n\n")

    def _append_chat(self, text):
        self.ui_queue.put(("text", (self.chat_display, text)))

    # ================= 右侧文档管理 =================

    def refresh_file_list(self):
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)

        if not DEFAULT_OUTPUT_DIR.exists(): return

        valid_extensions = {".md", ".html", ".txt"}
        for p in DEFAULT_OUTPUT_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in valid_extensions:
                size_kb = f"{max(1, round(p.stat().st_size / 1024)):,} KB"
                self.file_tree.insert("", tk.END, iid=str(p), values=(p.name, size_kb))

    def open_selected_file(self):
        selected = self.file_tree.selection()
        if not selected: return
        try:
            if os.name == 'nt':
                os.startfile(selected[0])
            elif os.name == 'posix':
                import subprocess
                subprocess.call(('xdg-open', selected[0]))
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件:\n{e}")

    def delete_selected_file(self):
        selected = self.file_tree.selection()
        if not selected: return
        filepath = Path(selected[0])
        if messagebox.askyesno("确认", f"确定删除文档 {filepath.name} 吗？"):
            try:
                if filepath.exists():
                    filepath.unlink()
                self.refresh_file_list()

                # 如果 ChromaDB 支持按 metadata source 删除，可以加在这里。这里从简。
                print(f"🗑️ 删除文件: {filepath.name}")
            except Exception as e:
                messagebox.showerror("错误", f"删除失败:\n{e}")

    def open_output_dir(self):
        try:
             if os.name == 'nt':
                 os.startfile(str(DEFAULT_OUTPUT_DIR))
             elif os.name == 'posix':
                 import subprocess
                 subprocess.call(('xdg-open', str(DEFAULT_OUTPUT_DIR)))
        except Exception as e:
            pass


if __name__ == "__main__":
    app = ScraplingApp()
    app.mainloop()
