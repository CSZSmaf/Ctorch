"use strict";

const byId = id => document.getElementById(id);
const lessons = [...document.querySelectorAll(".lesson")];
const courseLinks = [...document.querySelectorAll("#lesson-nav a")];
const completed = new Set(JSON.parse(localStorage.getItem("ctorch-lessons") || "[]"));
const searchIndex = lessons.map(lesson => ({
  id: lesson.id,
  title: lesson.dataset.title,
  description: lesson.dataset.description,
  text: lesson.textContent.replace(/\s+/g, " ")
}));
let currentLesson;
let trainingTimer;
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
  byId("menu").setAttribute("aria-label", open ? "关闭课程目录" : "打开课程目录");
  byId("scrim").hidden = !open;
}
byId("menu").onclick = () => setMenu(!document.body.classList.contains("menu-open"));
byId("scrim").onclick = () => setMenu(false);

function updateProgress() {
  const count = lessons.filter(lesson => completed.has(lesson.id)).length;
  byId("progress").value = count;
  byId("progress-text").textContent = `${count} / ${lessons.length}`;
  courseLinks.forEach(link => {
    link.querySelector(".done-mark").textContent = completed.has(link.hash.slice(1)) ? "✓" : "";
  });
  const done = completed.has(currentLesson.id);
  byId("complete").textContent = done ? "✓ 本节已完成" : "标记本节已完成";
  byId("complete").setAttribute("aria-pressed", String(done));
  byId("completion-note").textContent = done ? "可以继续下一节，也可以随时回来复习。" : "按自己的节奏继续。";
}

// 正文直接存在于 HTML 中；路由只选择当前课程，并从其标题生成页内目录。
function navigate() {
  const target = byId(decodeURIComponent(location.hash.slice(1))) || lessons[0];
  const lesson = target.closest(".lesson") || currentLesson || lessons[0];
  const changed = lesson !== currentLesson;
  const moveFocus = currentLesson && changed;
  lessons.forEach(item => { item.hidden = item !== lesson; });
  currentLesson = lesson;
  courseLinks.forEach(link => {
    if (link.hash === `#${lesson.id}`) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  byId("breadcrumb-title").textContent = lesson.dataset.title;
  document.title = `${lesson.dataset.title} · Ctorch 教程`;
  byId("page-toc").replaceChildren(...[...lesson.querySelectorAll("h2")].map(heading => {
    const link = document.createElement("a");
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    return link;
  }));
  const index = lessons.indexOf(lesson);
  [["previous", index - 1], ["next", index + 1]].forEach(([id, position]) => {
    const link = byId(id);
    link.hidden = position < 0 || position >= lessons.length;
    if (!link.hidden) {
      link.href = `#${lessons[position].id}`;
      link.querySelector("strong").textContent = lessons[position].dataset.title;
    }
  });
  setMenu(false);
  if (changed) stopTraining();
  updateProgress();
  if (moveFocus) byId("content").focus({ preventScroll: true });
  requestAnimationFrame(() => {
    if (target !== lesson && target.closest(".lesson")) target.scrollIntoView();
    else if (changed) window.scrollTo(0, 0);
    updateToc();
  });
}
function updateToc() {
  const headings = [...currentLesson.querySelectorAll("h2")];
  const active = headings.filter(heading => heading.getBoundingClientRect().top < 160).at(-1) || headings[0];
  document.querySelectorAll("#page-toc a").forEach(link => link.classList.toggle("active", link.hash === `#${active.id}`));
}
window.addEventListener("hashchange", navigate);
window.addEventListener("scroll", updateToc, { passive: true });
byId("complete").onclick = () => {
  if (completed.has(currentLesson.id)) completed.delete(currentLesson.id);
  else completed.add(currentLesson.id);
  localStorage.setItem("ctorch-lessons", JSON.stringify([...completed]));
  updateProgress();
};
byId("print").onclick = () => window.print();

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

function search() {
  const query = byId("search-input").value.trim().toLowerCase();
  const matches = searchIndex.filter(entry => entry.text.toLowerCase().includes(query));
  byId("search-results").replaceChildren(...matches.map(entry => {
    const link = document.createElement("a");
    const title = document.createElement("strong");
    const description = document.createElement("p");
    link.href = `#${entry.id}`;
    title.textContent = entry.title;
    const position = entry.text.toLowerCase().indexOf(query);
    description.textContent = query ? `…${entry.text.slice(Math.max(0, position - 25), position + 100)}…` : entry.description;
    link.append(title, description);
    link.onclick = () => byId("search-dialog").close();
    return link;
  }));
  if (matches.length === 0) {
    const empty = document.createElement("p");
    empty.className = "search-empty";
    empty.textContent = "没有匹配的课程。可以换一个关键词，例如“梯度”或“状态”。";
    byId("search-results").append(empty);
  }
}
function openSearch() {
  byId("search-dialog").showModal();
  byId("search-input").value = "";
  search();
  byId("search-input").focus();
}
byId("search-open").onclick = openSearch;
byId("search-close").onclick = () => byId("search-dialog").close();
byId("search-input").oninput = search;
byId("search-input").onkeydown = event => {
  const first = byId("search-results").querySelector("a");
  if (event.key === "ArrowDown" && first) { event.preventDefault(); first.focus(); }
  if (event.key === "Enter" && first) first.click();
};
document.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (byId("search-dialog").open) byId("search-dialog").close();
    else openSearch();
  }
  if (event.key === "Escape") {
    if (byId("search-dialog").open) {
      event.preventDefault();
      byId("search-dialog").close();
    }
    setMenu(false);
  }
});

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
    const download = document.createElement("a");
    download.href = pre.dataset.source;
    download.download = "";
    download.textContent = "下载完整示例 ↓";
    toolbar.append(download);
  }
  const copy = document.createElement("button");
  copy.textContent = "复制代码";
  copy.onclick = async () => {
    await navigator.clipboard.writeText(source);
    toast("代码已复制");
  };
  toolbar.append(copy);
  pre.before(toolbar);
  if (!pre.dataset.language.startsWith("C++")) return;
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

const tabs = [...document.querySelectorAll('[role="tab"]')];
tabs.forEach((tab, index) => {
  tab.onclick = () => tabs.forEach(item => {
    const active = item === tab;
    item.setAttribute("aria-selected", String(active));
    item.tabIndex = active ? 0 : -1;
    byId(item.getAttribute("aria-controls")).hidden = !active;
  });
  tab.onkeydown = event => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const next = tabs[(index + 1) % tabs.length];
      next.click();
      next.focus();
    }
  };
});

const stages = [
  ["auto input = dl::from_span<float>(x, {5, 1});", "把五个样本放入一个批次。shape 的第一维是样本数，第二维是每个样本的特征数。"],
  ["auto prediction = model(input);", "Linear 使用当前权重和偏置计算预测。前向中的张量操作同时记录反向所需的依赖。"],
  ["auto loss = dl::mse_loss(prediction, target);", "均方误差把每个预测与目标的偏差平方，再对所有元素取平均，得到一个标量。"],
  ["loss.backward();", "沿依赖反向计算每个参数的梯度。共享参数在多条路径上的贡献会汇总到同一叶梯度。"],
  ["optimizer.step();", "SGD 沿负梯度方向更新参数。下一轮先 zero_grad，再用新参数重复这段过程。"]
];
document.querySelectorAll("[data-stage]").forEach(button => {
  button.onclick = () => {
    document.querySelectorAll("[data-stage]").forEach(item => {
      item.classList.toggle("selected", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    const [code, text] = stages[Number(button.dataset.stage)];
    byId("workflow-code").textContent = code;
    byId("workflow-text").textContent = text;
  };
});

let transposed = false;
let selectedOffset = 0;
function selectTensorCell() {
  byId("tensor-grid").querySelectorAll("button").forEach(button => {
    const active = Number(button.dataset.offset) === selectedOffset;
    button.classList.toggle("selected", active);
    button.setAttribute("aria-pressed", String(active));
  });
  [...byId("storage-strip").children].forEach((cell, i) => cell.classList.toggle("selected", i === selectedOffset));
  const row = transposed ? selectedOffset % 3 : Math.floor(selectedOffset / 3);
  const col = transposed ? Math.floor(selectedOffset / 3) : selectedOffset % 3;
  byId("tensor-address").textContent = `offset(${row}, ${col}) = ${row} × ${transposed ? 1 : 3} + ${col} × ${transposed ? 3 : 1} = ${selectedOffset}`;
}
function renderTensor() {
  const rows = transposed ? 3 : 2, cols = transposed ? 2 : 3;
  byId("tensor-grid").style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  byId("tensor-grid").replaceChildren(...Array.from({ length: 6 }, (_, i) => {
    const row = Math.floor(i / cols), col = i % cols;
    const offset = transposed ? row + col * 3 : row * 3 + col;
    const button = document.createElement("button");
    button.dataset.offset = offset;
    button.innerHTML = `${offset + 1}<small>存储 [${offset}]</small>`;
    button.setAttribute("aria-label", `第 ${row} 行，第 ${col} 列，值 ${offset + 1}，存储下标 ${offset}`);
    button.onclick = () => { selectedOffset = offset; selectTensorCell(); };
    return button;
  }));
  byId("storage-strip").innerHTML = Array.from({ length: 6 }, (_, i) => `<span>${i + 1}<small>[${i}]</small></span>`).join("");
  byId("tensor-shape").textContent = `shape = [${rows}, ${cols}]`;
  byId("tensor-stride").textContent = transposed ? "stride = [1, 3]" : "stride = [3, 1]";
  byId("transpose").textContent = transposed ? "恢复原矩阵" : "转置矩阵";
  byId("transpose").setAttribute("aria-pressed", String(transposed));
  selectTensorCell();
}
byId("transpose").onclick = () => { transposed = !transposed; renderTensor(); };
renderTensor();

let accumulatedGradient = null;
function renderGradient() {
  const x = Number(byId("gradient-x").value);
  byId("gradient-x-value").textContent = byId("graph-x").textContent = x.toFixed(1);
  byId("gradient-square").textContent = (2 * x).toFixed(1);
  byId("graph-loss").textContent = (x * x + x).toFixed(2);
  byId("gradient-result").textContent = (2 * x + 1).toFixed(1);
  byId("accumulated-gradient").textContent = accumulatedGradient === null ? "空" : accumulatedGradient.toFixed(1);
}
byId("gradient-x").oninput = () => { accumulatedGradient = null; renderGradient(); };
byId("backward").onclick = () => {
  accumulatedGradient = (accumulatedGradient || 0) + 2 * Number(byId("gradient-x").value) + 1;
  renderGradient();
};
byId("zero-gradient").onclick = () => { accumulatedGradient = null; renderGradient(); };
renderGradient();

// 五个样本与 first_training.cpp 相同；两个偏导由同一批次的 MSE 计算。
const samples = [-2, -1, 0, 1, 2];
let weight = 0, bias = 0, updates = 0, losses = [9];
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const mse = () => mean(samples.map(x => (weight * x + bias - (2 * x + 1)) ** 2));
const chartLine = (x1, y1, x2, y2, kind) => `<line class="${kind}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
function renderTraining() {
  byId("train-step").textContent = updates;
  byId("train-weight").textContent = weight.toFixed(4);
  byId("train-bias").textContent = bias.toFixed(4);
  const loss = losses.at(-1);
  byId("train-loss").textContent = loss > 0 && loss < .0001 ? loss.toExponential(2) : loss.toFixed(4);
  const px = x => 42 + (x + 2.5) / 5 * 340;
  const py = y => 218 - (y + 5) / 12 * 200;
  const axes = [-4, 0, 4].map(y => chartLine(42, py(y), 382, py(y), "chart-grid") + `<text x="31" y="${py(y) + 4}" text-anchor="end">${y}</text>`).join("")
    + [-2, 0, 2].map(x => `<text x="${px(x)}" y="240" text-anchor="middle">${x}</text>`).join("")
    + chartLine(42, py(0), 382, py(0), "chart-axis");
  byId("fit-chart").innerHTML = `<defs><clipPath id="fit-clip"><rect x="42" y="18" width="340" height="200"/></clipPath></defs>${axes}<g clip-path="url(#fit-clip)">${chartLine(px(-2.5), py(weight * -2.5 + bias), px(2.5), py(weight * 2.5 + bias), "prediction")}${samples.map(x => `<circle class="data-point" cx="${px(x)}" cy="${py(2 * x + 1)}" r="6"/>`).join("")}</g>`;
  const maxStep = Math.max(50, updates);
  const lx = step => 54 + step / maxStep * 328;
  const ly = value => 218 - (Math.log10(Math.max(value, 1e-12)) + 12) / 13 * 200;
  const lossGrid = [1, -3, -7, -11].map(power => chartLine(54, ly(10 ** power), 382, ly(10 ** power), "chart-grid") + `<text x="45" y="${ly(10 ** power) + 4}" text-anchor="end">1e${power}</text>`).join("");
  byId("loss-chart").innerHTML = `<title>损失的十进制对数坐标；低于 1e-12 的值显示在底部。</title>${lossGrid}<path class="loss-line" d="${losses.map((value, i) => `${i ? "L" : "M"}${lx(i)},${ly(value)}`).join(" ")}"/><circle cx="${lx(updates)}" cy="${ly(loss)}" r="3" fill="var(--green)"/><text x="54" y="240">0</text><text x="382" y="240" text-anchor="end">${maxStep} 步</text>`;
}
function trainStep() {
  const errors = samples.map(x => weight * x + bias - (2 * x + 1));
  const dw = 2 * mean(errors.map((error, i) => error * samples[i]));
  const db = 2 * mean(errors);
  const rate = Number(byId("learning-rate").value);
  weight -= rate * dw;
  bias -= rate * db;
  ++updates;
  losses.push(mse());
  byId("train-status").textContent = `已更新 ${updates} 步，η = ${rate.toFixed(2)}。`;
  renderTraining();
}
function stopTraining() {
  clearInterval(trainingTimer);
  trainingTimer = null;
  byId("train-many").textContent = "连续 50 步";
  byId("train-one").disabled = false;
}
byId("train-one").onclick = trainStep;
byId("train-many").onclick = () => {
  if (trainingTimer) { stopTraining(); return; }
  const end = updates + 50;
  byId("train-many").textContent = "暂停";
  byId("train-one").disabled = true;
  trainingTimer = setInterval(() => { trainStep(); if (updates === end) stopTraining(); }, 65);
};
byId("train-reset").onclick = () => {
  stopTraining();
  weight = bias = updates = 0;
  losses = [9];
  byId("train-status").textContent = "从 w = 0、b = 0 开始。";
  renderTraining();
};
byId("learning-rate").oninput = () => { byId("learning-rate-value").textContent = Number(byId("learning-rate").value).toFixed(2); };
renderTraining();

document.querySelectorAll("[data-binding]").forEach(button => {
  button.onclick = () => {
    const external = button.dataset.binding === "functional";
    document.querySelectorAll("[data-binding]").forEach(item => {
      item.classList.toggle("selected", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    byId("binding-result").textContent = external ? "15" : "6";
    byId("binding-text").textContent = external ? "本次调用读取外部 weight = 5。调用结束后，模型自身的 weight 仍为 2。" : "普通调用读取模块登记的参数。外部状态的更改保留在自己的存储中。";
  };
});

let stateValue = 1, snapshotValue = 1;
function renderState() {
  byId("state-model").textContent = byId("state-view").textContent = stateValue;
  byId("state-snapshot-value").textContent = snapshotValue;
}
byId("state-update").onclick = () => { ++stateValue; renderState(); };
byId("state-capture").onclick = () => { snapshotValue = stateValue; renderState(); };
byId("state-reset").onclick = () => { stateValue = snapshotValue = 1; renderState(); };
byId("fusion").onchange = () => {
  const fused = byId("fusion").checked;
  byId("plan-operations").innerHTML = fused ? '<span class="operation fused">Add + ReLU</span>' : '<span class="operation">Add</span> → <span class="operation">ReLU</span>';
  byId("plan-steps").textContent = fused ? "1" : "2";
  byId("plan-fusions").textContent = fused ? "1" : "0";
};

navigate();
