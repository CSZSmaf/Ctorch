"use strict";

const framework = document.getElementById("comparison-framework");
const step = document.getElementById("comparison-step");
const chart = document.getElementById("comparison-chart");
const colors = {ctorch: "var(--accent)", pytorch: "var(--green)", tensorflow: "var(--blue)"};
const x = index => 48 + index / 199 * 680;
const y = value => 230 - value / 2.5 * 206;

// 每个点来自一次实际训练的批次损失，选择框架时保留相同的坐标范围。
function renderComparison() {
  const index = Number(step.value) - 1;
  const selected = framework.value;
  const names = selected === "all" ? Object.keys(comparisonRuns) : [selected];
  const grid = [0, 0.5, 1, 1.5, 2, 2.5].map(value =>
    `<line class="chart-grid" x1="48" x2="728" y1="${y(value)}" y2="${y(value)}"/><text x="38" y="${y(value) + 4}" text-anchor="end">${value.toFixed(1)}</text>`).join("");
  const curves = names.map(name => `<path d="${comparisonRuns[name].map((value, i) => `${i ? "L" : "M"}${x(i)},${y(value)}`).join(" ")}" fill="none" stroke="${colors[name]}" stroke-width="2"/>`).join("");
  chart.innerHTML = `<title>MNIST MLP 的 200 步训练批次损失</title>${grid}${curves}<line x1="${x(index)}" x2="${x(index)}" y1="24" y2="230" stroke="var(--ink)" stroke-dasharray="4 4"/><text x="48" y="253">1</text><text x="728" y="253" text-anchor="end">200 步</text>`;
  document.getElementById("comparison-step-value").textContent = step.value;
  for (const name of Object.keys(comparisonRuns))
    document.getElementById(`comparison-loss-${name}`).textContent = comparisonRuns[name][index].toFixed(6);
}

framework.addEventListener("change", renderComparison);
step.addEventListener("input", renderComparison);
renderComparison();
