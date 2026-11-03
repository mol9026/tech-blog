# 发布与分发 Checklist

> 每篇文章从写作到全网分发的固定流程。跑熟后可以把这个清单固化成 skill。

## 1. 写作

- [ ] 新文章目录：`content/posts/<slug>/index.md`（一文一目录，配图放旁边）
- [ ] front matter 必填：`title` / `date` / `slug`（短横线英文 slug，一旦发布不改）/ `tags` / `summary`
- [ ] `draft: true` 写作期；发布前改 `draft: false`
- [ ] 预览：`hugo server -D`，本地地址 http://localhost:1313/tech-blog/

## 2. 发布前检查（纪律项）

- [ ] **脱敏**：工号、内部容器名/机器名、客户信息、内部指标全部去除或泛化
- [ ] **对外口径**：数据可公开、结论留有余地、不点名内部团队
- [ ] 文中数字与原始记录核对过一遍
- [ ] 代码块可独立运行（或注明依赖）

## 3. 发布（T1：全自动）

```bash
export PATH=/c/vscodedev/tools/hugo-bin:$PATH
cd /c/vscodedev/tech-blog
hugo                # 本地构建验证
git add -A && git commit -m "post: <slug>" && git push
```

- Actions 自动构建部署到 https://mol9026.github.io/tech-blog/
- 发布后验证：首页 200、文章页 200、RSS `index.xml` 有新条目
- RSS 地址：`https://mol9026.github.io/tech-blog/index.xml`

## 4. 分发（T2：半自动，发布隔天执行）

自站首发 → 隔天分发，每篇分发稿**必须**带：

```
首发于：<原文 URL>
转载已获授权，内容可能未同步更新，建议访问原文。
```

- [ ] **掘金 / 知乎 / CSDN / 公众号**：用 wechatsync（浏览器插件）一键推送，每站检查一遍公式和代码块渲染
- [ ] **公众号**：mdnice（墨滴）转格式，代码块主题选跟站点一致的深色
- [ ] **CSDN**：也可走 MetaWeblog API 半自动
- [ ] 各分发稿评论区留原文链接，读者问题回流到自站

## 5. 收尾

- [ ] 若版本收敛/后续进展，回到原文更新并追加「更新记录」节
- [ ] 选题池：从 skill 仓库的 skills-catalog、排查记录里挑下一个坯子
