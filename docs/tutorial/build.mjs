import { mkdir, readFile, writeFile, copyFile, cp } from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve, join, dirname, posix } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const source = fileURLToPath(new URL(".", import.meta.url));
const root = resolve(source, "../..");
const destination = resolve(process.argv[2]);
const require = createRequire(resolve(process.argv[3], "package.json"));
const katex = require("katex");
const { Marked } = await import(pathToFileURL(require.resolve("marked")));
const template = await readFile(join(source, "reader.html"), "utf8");
const pages = new Map();
const images = new Map();
const sections = {
  tutorial: { title: "教程", pages: [
    ["docs/tutorial/index.html", "课程目录"],
    ["docs/tutorial/README.md", "教程使用"],
  ] },
  projects: { title: "项目", pages: [
    ["docs/projects/README.md", "项目目录"],
    ["docs/projects/mnist.md", "MNIST：MLP 与 LeNet"],
    ["docs/projects/resnet.md", "CIFAR-10：ResNet-20"],
    ["docs/projects/transformer.md", "Shakespeare：Transformer"],
    ["docs/projects/comparison.md", "框架结果对比"],
    ["docs/projects/performance.md", "多维性能比较"],
  ] },
  manual: { title: "手册", pages: [
    ["docs/README.md", "手册目录"],
    ["docs/build.md", "构建与安装"],
    ["docs/tensors.md", "张量与梯度"],
    ["docs/training.md", "模型与训练"],
    ["docs/execution.md", "组合与执行"],
    ["docs/performance.md", "性能与内存"],
    ["docs/extensions.md", "C17 与外部计算"],
    ["docs/npu.md", "OpenVINO / Intel NPU"],
    ["docs/architecture.md", "架构"],
    ["tests/README.md", "测试说明"],
  ] },
  examples: { title: "示例", pages: [
    ["examples/README.md", "示例目录"],
    ["examples/first_training.cpp", "第一次训练"],
    ["examples/training_basics.cpp", "批次、评估与恢复"],
    ["examples/parameter_layer.cpp", "自定义参数层"],
    ["examples/mnist_mlp.cpp", "MNIST MLP"],
    ["examples/mnist_lenet.cpp", "MNIST LeNet"],
    ["examples/cifar_resnet.cpp", "CIFAR-10 ResNet"],
    ["examples/shakespeare_transformer.cpp", "Shakespeare Transformer"],
    ["examples/text_generate.cpp", "文本生成"],
    ["examples/comparison/README.md", "跨框架比较"],
  ] },
};
const namedRoutes = new Map([
  ["docs/tutorial/index.html", "index.html"],
  ["README.md", "manual/overview.html"],
  ["docs/README.md", "manual/index.html"],
  ["docs/tutorial/README.md", "manual/tutorial.html"],
  ["examples/README.md", "examples/index.html"],
  ["examples/comparison/README.md", "examples/comparison/index.html"],
  ["tests/README.md", "manual/testing.html"],
  ["docs/projects/README.md", "projects/index.html"],
]);
const escape = value => value.replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const relative = (page, target) => posix.relative(posix.dirname(page), target);

// 链接同时登记待生成文件；手册与示例引用的文件会随网站一起发布。
function route(file) {
  const page = namedRoutes.get(file) || (file.startsWith("docs/projects/")
    ? file.replace(/^docs\//, "").replace(/\.md$/, ".html") : file.endsWith(".md")
    ? `manual/${posix.basename(file, ".md")}.html` : `source/${file}.html`);
  pages.set(file, page);
  return page;
}
function link(file, page, href) {
  if (/^(?:[a-z]+:|#|\/)/i.test(href)) return href;
  const [path, fragment = ""] = href.split("#");
  const target = posix.normalize(posix.join(posix.dirname(file), decodeURI(path)));
  return relative(page, route(target)) + (fragment ? `#${fragment}` : "");
}
function image(file, page, href) {
  if (/^(?:[a-z]+:|\/)/i.test(href)) return href;
  const target = posix.normalize(posix.join(posix.dirname(file), decodeURI(href)));
  const output = `assets/${target}`;
  images.set(target, output);
  return relative(page, output);
}
async function save(file, content) {
  const target = join(destination, file);
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, content);
}
function math(html) {
  return html.replace(/<span([^>]*?) data-tex="([^"]*)"([^>]*)>[\s\S]*?<\/span>/g,
    (_, before, tex, after) => `<span${before}${after}>${katex.renderToString(tex, {
      displayMode: (before + after).includes("data-display"),
      output: "htmlAndMathml",
    })}</span>`);
}
function reader(file, page, title, body, headings = []) {
  const prefix = relative(page, "index.html").replace(/index\.html$/, "");
  const section = file.startsWith("docs/projects/") ? sections.projects
    : file.startsWith("examples/") ? sections.examples
    : file.startsWith("docs/tutorial/") ? sections.tutorial : sections.manual;
  const navigation = section.pages.some(([target]) => target === file)
    ? section.pages : [...section.pages, [file, title]];
  const values = {
    TITLE: escape(title), PAGE: page, ROOT: prefix, CONTENT: math(body),
    LICENSE: relative(page, route("LICENSE")),
    SECTION: section.title,
    SECTION_HOME: relative(page, route(section.pages[0][0])),
    TOP_NAVIGATION: Object.values(sections).map(item =>
      `<a href="${relative(page, route(item.pages[0][0]))}"${item === section ? ' class="current" aria-current="true"' : ""}>${item.title}</a>`).join(""),
    NAVIGATION: navigation.map(([target, label]) =>
      `<a href="${relative(page, route(target))}"${target === file ? ' aria-current="page"' : ""}>${label}</a>`).join("\n"),
    TOC: headings.map(({ id, title, depth }) =>
      `<a href="#${id}" class="toc-depth-${depth}">${escape(title)}</a>`).join("\n"),
    SCRIPTS: file === "docs/projects/comparison.md"
      ? `<script src="${prefix}comparison-data.js" defer></script><script src="${prefix}comparison.js" defer></script>` : "",
  };
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => values[key]);
}

await mkdir(destination, { recursive: true });
const katexRoot = dirname(dirname(require.resolve("katex")));
await cp(join(katexRoot, "dist/fonts"), join(destination, "assets/katex/fonts"), { recursive: true });
await copyFile(join(katexRoot, "dist/katex.min.css"), join(destination, "assets/katex/katex.min.css"));
await copyFile(join(katexRoot, "LICENSE"), join(destination, "assets/katex/LICENSE"));
for (const file of ["style.css", "interface.js", "tutorial.js", "comparison-data.js", "comparison.js"]) {
  await copyFile(join(source, file), join(destination, file));
}
for (const file of ["LICENSE", "COPYRIGHT"]) {
  await copyFile(join(root, file), join(destination, file));
  route(file);
}

const main = "docs/tutorial/index.html";
const html = (await readFile(join(root, main), "utf8"))
  .replace(/(href|data-source)="(\.\.?\/[^\"]+)"/g,
    (_, attribute, href) => `${attribute}="${link(main, "index.html", href)}"`)
  .replace('<link rel="stylesheet" href="style.css">',
    '<link rel="stylesheet" href="assets/katex/katex.min.css">\n  <link rel="stylesheet" href="style.css">');
await save("index.html", math(html));
for (const section of Object.values(sections))
  for (const [file] of section.pages) route(file);

// 每份 Markdown 生成正文和标题目录，源文件页另附可下载的原文件。
for (const [file, page] of pages) {
  if (page === "index.html") continue;
  const text = await readFile(join(root, file), "utf8");
  if (file.endsWith(".md")) {
    let title;
    const headings = [];
    const ids = new Map();
    const markdown = new Marked({ renderer: {
      heading({ text, tokens, depth }) {
        const base = text.toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, "").trim().replace(/\s+/g, "-");
        const count = ids.get(base) || 0;
        ids.set(base, count + 1);
        const id = count ? `${base}-${count}` : base;
        if (depth === 1) title = text;
        else headings.push({ id, title: text, depth });
        return `<h${depth} id="${id}">${this.parser.parseInline(tokens)}</h${depth}>\n`;
      },
      link({ href, title, tokens }) {
        return `<a href="${escape(link(file, page, href))}"${title ? ` title="${escape(title)}"` : ""}>${this.parser.parseInline(tokens)}</a>`;
      },
      image({ href, title, text }) {
        return `<img src="${escape(image(file, page, href))}" alt="${escape(text)}"${title ? ` title="${escape(title)}"` : ""} loading="lazy" style="max-width:100%;height:auto">`;
      },
      code({ text, lang = "text" }) {
        const language = ({ cpp: "C++", c: "C17", cmake: "CMake", bash: "Shell", sh: "Shell" })[lang] || lang;
        return `<pre data-language="${escape(language)}"><code>${escape(text)}</code></pre>\n`;
      },
    } });
    const body = markdown.parse(text);
    await save(page, reader(file, page, title, body, headings));
  } else {
    const title = posix.basename(file);
    const language = file.endsWith(".py") ? "Python" : file.endsWith(".c") ? "C17"
      : /\.(cpp|h)$/.test(file) ? "C++" : "Text";
    const download = `downloads/${file}`;
    await save(download, text);
    const body = `<h1>${title}</h1><p><a href="${relative(page, download)}" download>下载 ${title}</a></p>
<pre data-language="${language}"><code>${escape(text)}</code></pre>`;
    await save(page, reader(file, page, title, body));
  }
}
for (const [file, output] of images) {
  await mkdir(dirname(join(destination, output)), { recursive: true });
  await copyFile(join(root, file), join(destination, output));
}
console.log(`Generated ${new Set(["index.html", ...pages.values()]).size} pages in ${destination}`);
