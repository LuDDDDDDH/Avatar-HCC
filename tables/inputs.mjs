import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const position = process.argv.indexOf('--config');
const filename = position >= 0 ? process.argv[position + 1] : process.env.AVATAR_TABLE_CONFIG;
if (!filename) throw new Error('Set AVATAR_TABLE_CONFIG or provide --config');
const configFile = path.resolve(filename);
const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
const base = path.dirname(configFile);
const codeRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
if (!config.workspace) throw new Error('Missing workspace');
const workspace = path.resolve(base, config.workspace);
const inside = (parent, child) => {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
};
if (inside(codeRoot, workspace) || inside(workspace, codeRoot)) throw new Error('Workspace must be separate from source');

function required(section, key) {
  const value = config[section]?.[key];
  if (typeof value !== 'string' || value.length === 0) throw new Error(`Missing ${section}.${key}`);
  return value;
}

export function inputPath(key) {
  return path.resolve(base, required('files', key));
}

export function outputPath(key) {
  const result = path.resolve(base, required('files', key));
  if (!inside(workspace, result) || inside(codeRoot, result)) throw new Error(`Invalid output ${key}`);
  if (fs.existsSync(result)) throw new Error(`Output already exists: ${key}`);
  return result;
}

export function outputDirectory(key) {
  const result = path.resolve(base, required('directories', key));
  if (!inside(workspace, result) || inside(codeRoot, result)) throw new Error(`Invalid directory ${key}`);
  return result;
}

export function label(key) {
  return required('labels', key);
}
