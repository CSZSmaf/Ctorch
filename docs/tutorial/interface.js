"use strict";

const byId = id => document.getElementById(id);
let toastTimer;

function toast(message) {
  byId("toast").textContent = message;
  byId("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { byId("toast").hidden = true; }, 2200);
}

function setMenu(open) {
  document.body.classList.toggle("menu-open", open);
  byId("menu").setAttribute("aria-expanded", String(open));
  byId("menu").setAttribute("aria-label", open ? "关闭目录" : "打开目录");
  byId("scrim").hidden = !open;
}
byId("menu").onclick = () => setMenu(!document.body.classList.contains("menu-open"));
byId("scrim").onclick = () => setMenu(false);

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  byId("theme").setAttribute("aria-label", theme === "dark" ? "切换浅色主题" : "切换深色主题");
}
setTheme(localStorage.getItem("ctorch-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
byId("theme").onclick = () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  setTheme(theme);
  localStorage.setItem("ctorch-theme", theme);
};

// 代码以纯文本保存，着色仅包裹匹配的词法片段，复制始终取得原始文本。
document.querySelectorAll("pre[data-language]").forEach(pre => {
  const code = pre.querySelector("code");
  const source = code.textContent;
  const toolbar = document.createElement("div");
  toolbar.className = "code-toolbar";
  const language = document.createElement("span");
  language.textContent = pre.dataset.language;
  toolbar.append(language);
  if (pre.dataset.source) {
    const example = document.createElement("a");
    example.href = pre.dataset.source;
    example.textContent = "查看完整示例";
    toolbar.append(example);
  }
  const copy = document.createElement("button");
  copy.textContent = "复制代码";
  copy.onclick = async () => {
    await navigator.clipboard.writeText(source);
    toast("代码已复制");
  };
  toolbar.append(copy);
  pre.before(toolbar);
  if (!/^C(\+\+|17)/.test(pre.dataset.language)) return;
  const tokens = /\/\/[^\n]*|"(?:\\.|[^"\\])*"|\b(?:auto|const|class|public|private|return|void|int|float|bool|for|if|else|true|false|override|include|namespace)\b/g;
  let offset = 0;
  const fragment = document.createDocumentFragment();
  for (const match of source.matchAll(tokens)) {
    fragment.append(document.createTextNode(source.slice(offset, match.index)));
    const span = document.createElement("span");
    span.className = match[0].startsWith("//") ? "syntax-comment" : match[0].startsWith('"') ? "syntax-string" : "syntax-keyword";
    span.textContent = match[0];
    fragment.append(span);
    offset = match.index + match[0].length;
  }
  fragment.append(document.createTextNode(source.slice(offset)));
  code.replaceChildren(fragment);
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape") setMenu(false);
});
