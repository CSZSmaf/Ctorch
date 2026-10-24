import { mkdir, readFile, writeFile, copyFile } from "node:fs/promises";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";

const source = fileURLToPath(new URL(".", import.meta.url));
const destination = resolve(process.argv[2]);
const repository = "https://github.com/CSZSmaf/Ctorch/blob/main/docs/tutorial/";
await mkdir(destination, { recursive: true });

// 在线手册和源码由 GitHub 展示；原始网页保留相对链接，供离线阅读。
const html = (await readFile(join(source, "index.html"), "utf8")).replace(
  /(href|data-source)="(\.\.?\/[^\"]+)"/g,
  (_, attribute, path) => `${attribute}="${new URL(path, repository).href}"`,
);
await writeFile(join(destination, "index.html"), html);
for (const file of ["style.css", "tutorial.js"]) {
  await copyFile(join(source, file), join(destination, file));
}
