import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const root = path.resolve(__dirname, "..");
const source = path.resolve(root, "..", "outputs", "all_tests_summary.csv");
const sourceDir = path.dirname(source);
const targetDir = path.resolve(root, "public", "data");
const target = path.resolve(targetDir, "all_tests_summary.csv");
const once = process.argv.includes("--once");

const ensureTarget = () => {
  fs.mkdirSync(targetDir, { recursive: true });
};

const copyFile = () => {
  ensureTarget();
  if (!fs.existsSync(source)) {
    fs.writeFileSync(target, "scenario,protocol\n", "utf8");
    console.warn(`[sync-data] Source not found: ${source}`);
    return;
  }
  fs.copyFileSync(source, target);
  console.log(`[sync-data] Synced ${source} -> ${target}`);
};

copyFile();

if (!once) {
  try {
    fs.mkdirSync(sourceDir, { recursive: true });
  } catch (error) {
    console.warn(
      `[sync-data] Unable to prepare source directory ${sourceDir}: ${error.message}`
    );
    console.warn("[sync-data] Skipping watch; continuing without live sync.");
    return;
  }
  let timer = null;
  try {
    fs.watch(sourceDir, (eventType, filename) => {
      if (!filename || filename !== path.basename(source)) {
        return;
      }
      if (timer) {
        clearTimeout(timer);
      }
      timer = setTimeout(copyFile, 100);
    });
  } catch (error) {
    console.warn(
      `[sync-data] Unable to watch source directory ${sourceDir}: ${error.message}`
    );
  }
}
