// Frontend page smoke test: renders each built page in jsdom with a stubbed
// API and asserts the client-side scripts populate the DOM correctly.
//
// Run after `npm run build`:
//   npm run test:pages
//
// This mirrors the browser's deferred module-script behavior: scripts are
// evaluated only after the DOM is fully parsed.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(__dirname, '..', 'dist');

/**
 * Make a built ES module evaluable via `window.eval` by inlining its
 * `import {a, b as c} from "./chunk.js"` statements. The chunk is wrapped in an
 * IIFE that returns its exports as an object, and the import becomes a
 * destructure of that object — so the chunk's internal (mangled) names stay
 * scoped inside the IIFE and cannot collide with the page's top-level names.
 *
 * This mirrors how Vite names exports (e.g. `export{p as x}`) without needing a
 * real module loader or fighting Node's ESM cache.
 */
function inlineImports(content, baseDir) {
  const importRe = /import\s*\{([^}]*)\}\s*from\s*"([^"]+)";?/g;
  return content.replace(importRe, (full, specList, src) => {
    if (!src.startsWith('.')) return full; // leave bare specifiers untouched
    const chunkAbs = path.resolve(baseDir, src);
    let chunk = fs.readFileSync(chunkAbs, 'utf8');
    chunk = inlineImports(chunk, path.dirname(chunkAbs));
    const exportMap = {};
    chunk = chunk.replace(/export\s*\{([^}]*)\}\s*;?/g, (_ef, list) => {
      for (const pair of list.split(',').map((s) => s.trim()).filter(Boolean)) {
        const [internal, exported] = pair.split(/\s+as\s+/).map((s) => s.trim());
        exportMap[exported || internal] = internal;
      }
      return '';
    });
    const returnObj = '{' + Object.entries(exportMap).map(([e, i]) => `${e}:${i}`).join(',') + '}';
    const destructure = '{' + specList.split(',').map((s) => s.trim()).filter(Boolean).map((spec) => {
      const [imported, local] = spec.split(/\s+as\s+/).map((s) => s.trim());
      return local ? `${imported}:${local}` : imported;
    }).join(',') + '}';
    return `const ${destructure}=(function(){${chunk}return ${returnObj};})();`;
  });
}

const API = {
  '/api/models': {
    models: [
      { name: 'llama3.2:latest', host: 'lab-server', max_context: 131072, supports_vision: false, supports_tools: true },
      { name: 'qwen3.5:0.8b', host: 'lab-server', max_context: 262144, supports_vision: true, supports_tools: true },
    ],
    count: 2,
  },
  '/api/plugins': {
    plugins: [
      { id: 'smoke', name: 'Smoke / generation', description: 'Fast sanity check.', category: 'reasoning', version: '0.1.0', dataset_version: 'v1', modalities: ['text'], options: {} },
      { id: 'keyword', name: 'Keyword presence', description: 'Checks for expected keywords.', category: 'reasoning', version: '0.1.0', dataset_version: 'v1', modalities: ['text'], options: { max_hits: 3 } },
    ],
    compare_default: [],
  },
  '/api/benchmarks': {
    runs: [{ run_id: 'r1', timestamp: '2026-01-01T00:00:00Z', app_version: '0.1.0', model_names: ['m1'], hosts: [{ name: 'lab-server' }] }],
  },
  '/api/history': {
    runs: [{ run_id: 'r1', timestamp: '2026-01-01T00:00:00Z', app_version: '0.1.0', model_names: ['m1'], hosts: [{ name: 'lab-server' }] }],
    filters: { models: ['m1'], hosts: ['lab-server'] },
  },
  '/api/compare': {
    models: [{ model_name: 'm1', overall_score: 0.9, latency_p50_ms: 10, latency_p95_ms: 20, time_to_first_token_p50_ms: 5, tokens_per_second: 50, cases_run: 2, errors: 0 }],
  },
  '/api/compare?run=r1': {
    models: [
      { model_name: 'm1', run_id: 'r1', overall_score: 0.9,
        latency_p50_ms: 10, latency_p95_ms: 20, time_to_first_token_p50_ms: 5,
        tokens_per_second: 50, cases_run: 2, errors: 0,
        plugins: [
          { plugin_id: 'smoke', score: 0.9, latency_p50_ms: 10, time_to_first_token_p50_ms: 5, tokens_per_second: 50, cases_run: 2 },
        ] },
    ],
  },
  '/api/compare?run=r1&run=r2': {
    models: [
      { model_name: 'm1', run_id: 'r1', overall_score: 0.9, latency_p50_ms: 10, latency_p95_ms: 20, time_to_first_token_p50_ms: 5, tokens_per_second: 50, cases_run: 2, errors: 0, plugins: [{ plugin_id: 'smoke', score: 0.9, latency_p50_ms: 10, time_to_first_token_p50_ms: 5, tokens_per_second: 50, cases_run: 2 }] },
      { model_name: 'm2', run_id: 'r2', overall_score: 0.8, latency_p50_ms: 30, latency_p95_ms: 40, time_to_first_token_p50_ms: 15, tokens_per_second: 20, cases_run: 2, errors: 0, plugins: [{ plugin_id: 'smoke', score: 0.8, latency_p50_ms: 30, time_to_first_token_p50_ms: 15, tokens_per_second: 20, cases_run: 2 }] },
    ],
  },
  '/api/benchmarks/r1': {
    run: { run_id: 'r1' },
    models: [{ model_name: 'm1', overall_score: 0.9, latency_p50_ms: 10, latency_p95_ms: 20, time_to_first_token_p50_ms: 5, tokens_per_second: 50, cases_run: 2, errors: 0, plugins: [{ plugin_id: 'smoke', score: 0.9, latency_p50_ms: 10, time_to_first_token_p50_ms: 5, tokens_per_second: 50, cases_run: 2 }] }],
  },
  '/api/benchmarks/r1/cases': {
    run_id: 'r1', count: 1,
    cases: [{ model_name: 'm1', plugin_id: 'smoke', case_id: 'c1', passed: 1, score: 1.0, error: null, total_ms: 10, time_to_first_token_ms: 5, tokens_per_second: 50, prompt_tokens: 3, completion_tokens: 5, attempt: 1 }],
  },
  '/api/benchmarks/r2': {
    run: { run_id: 'r2' },
    models: [{ model_name: 'm2', overall_score: 0.8, latency_p50_ms: 30, latency_p95_ms: 40, time_to_first_token_p50_ms: 15, tokens_per_second: 20, cases_run: 2, errors: 0, plugins: [{ plugin_id: 'smoke', score: 0.8, latency_p50_ms: 30, time_to_first_token_p50_ms: 15, tokens_per_second: 20, cases_run: 2 }] }],
  },
  '/api/benchmarks/active': {
    runs: [
      {
        type: 'status', run_id: 'running1', status: 'running', progress: 0.5,
        total: 4, completed: 2, errors: 0, model: 'm1', plugin: 'smoke',
        models: ['m1'], plugins: ['smoke'], started_at: '2026-01-01T00:00:00Z',
      },
    ],
  },
  '/api/benchmarks/running1/status': {
    type: 'progress', run_id: 'running1', status: 'running', progress: 0.5,
    total: 4, completed: 2, errors: 0, model: 'm1', plugin: 'smoke', case_id: 'c1', message: null,
  },
  '/api/benchmarks/done1/status': {
    type: 'progress', run_id: 'done1', status: 'completed', progress: 1,
    total: 1, completed: 1, errors: 0, model: null, plugin: null, case_id: null, message: 'Benchmark completed',
  },
};

const NOT_FOUND = new Set(['/api/benchmarks/gone1/status', '/api/benchmarks/gone1']);

function loadPage(relPath, url, opts = {}) {
  const raw = fs.readFileSync(path.join(DIST, relPath), 'utf8');
  const scripts = [];
  // Astro inlines small module scripts but emits larger ones as external
  // files under /_astro/. Handle both, preserving document order.
  const html = raw.replace(
    /<script type="module"(?:\s+src="([^"]*)")?>([\s\S]*?)<\/script>/g,
    (_, src, body) => {
      if (src) {
        const file = path.join(DIST, src.replace(/^\//, ''));
        scripts.push(inlineImports(fs.readFileSync(file, 'utf8'), path.dirname(file)));
      } else {
        scripts.push(inlineImports(body, DIST));
      }
      return '';
    }
  );
  const dom = new JSDOM(html, {
    url: url || `http://localhost:8000/${relPath}`,
    runScripts: 'outside-only',
    beforeParse(window) {
       window.fetch = async (input) => {
         const key = typeof input === 'string' ? input : input.url;
         if (key in API) return { ok: true, json: async () => API[key] };
         if (NOT_FOUND.has(key)) return { ok: false, status: 404, json: async () => ({ detail: 'not found' }) };
         console.error('DEBUG fetch unmatched:', key);
         return { ok: false, json: async () => ({}) };
       };
       window.WebSocket = class { constructor() {} close() {} onmessage = null; onerror = null; };
       window.alert = () => {};
       window.confirm = () => true;
       window.__intervals = [];
       window.setInterval = (fn) => { window.__intervals.push(fn); return window.__intervals.length; };
       window.clearInterval = () => {};
    },
  });
  for (const [key, value] of Object.entries(opts.storage || {})) {
    dom.window.localStorage.setItem(key, value);
  }
  for (const body of scripts) {
    try {
      dom.window.eval(body);
    } catch (e) {
      dom.window.document.title = `SCRIPT ERROR: ${e.message}`;
    }
  }
  return dom.window.document;
}

function assert(name, cond) {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}`);
  if (!cond) process.exitCode = 1;
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const results = await Promise.all([
  loadPage('index.html'),
  loadPage('plugins/index.html'),
  loadPage('history/index.html'),
  loadPage('compare/index.html', 'http://localhost:8000/compare/'),
  loadPage('compare/index.html', 'http://localhost:8000/compare/?run=r1'),
  loadPage('compare/index.html', 'http://localhost:8000/compare/?run=r1&run=r2'),
  loadPage('run/index.html', 'http://localhost:8000/run/'),
  loadPage('run/index.html', 'http://localhost:8000/run/', { storage: { 'local-ai-bench.active-run': 'running1' } }),
  loadPage('run/index.html', 'http://localhost:8000/run/', { storage: { 'local-ai-bench.active-run': 'done1' } }),
  loadPage('run/index.html', 'http://localhost:8000/run/', { storage: { 'local-ai-bench.active-run': 'gone1' } }),
  loadPage('run/index.html', 'http://localhost:8000/run/?run=running1'),
]);
await wait(150);

const [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11] = results;

assert('index: stat-models = 2', d1.getElementById('stat-models').textContent === '2');
assert('index: stat-runs = 1', d1.getElementById('stat-runs').textContent === '1');
assert('index: last run not Never', d1.getElementById('stat-last').textContent !== 'Never');
assert('index: model card rendered', d1.getElementById('models').textContent.includes('llama3.2:latest'));
assert('index: vision badge', d1.getElementById('models').textContent.includes('Vision'));
assert('index: runs table', d1.getElementById('runs').textContent.includes('r1'));
assert('index: delete button present', d1.querySelector('button[data-delete]') !== null);
assert('index: active run shown', d1.getElementById('active-runs').textContent.includes('running1'));
assert('index: active run track link', d1.getElementById('active-runs').textContent.includes('Track'));

assert('plugins: cards rendered', d2.getElementById('plugins').textContent.includes('Keyword presence'));
assert('plugins: description shown', d2.getElementById('plugins').textContent.includes('Fast sanity check'));
assert('plugins: options editor present', d2.querySelector('#plugins form[data-plugin]') !== null);
assert('plugins: code buttons present', d2.querySelectorAll('button[data-source]').length >= 2);

assert('history: table with run', d3.getElementById('history').textContent.includes('r1'));
assert('history: delete button present', d3.querySelector('button[data-delete]') !== null);
assert('history: filter bar present', d3.getElementById('f-search') !== null && d3.getElementById('f-model') !== null);
assert('history: row checkboxes present', d3.querySelector('.row-check') !== null);
assert('history: select-all checkbox present', d3.querySelector('#select-all') !== null);
assert('history: batch delete button present', d3.querySelector('#bulk-delete') !== null);
assert('history: compare selected button present', d3.querySelector('#bulk-compare') !== null);
assert('history: compare per-row link removed', d3.querySelector('a[href^="/compare?run="]') === null);
assert('history: active run merged', d3.getElementById('history').textContent.includes('running1'));

assert('compare: column panel present', d4.getElementById('columns-popover') !== null);
assert('compare: model row', d4.getElementById('compare').textContent.includes('m1'));
assert('compare: score shown', d4.getElementById('compare').textContent.includes('0.900'));
assert('compare: tooltip glyph is i', d5.querySelector('th[title]') !== null && !d5.querySelector('th[title]').textContent.includes('?'));
assert('compare?run=r1: delete run button present', d5.querySelector('button[data-delete-run]') !== null);
assert('compare: column toggle checkboxes present', d5.querySelectorAll('input[data-col]').length >= 9);
assert('compare: minimal defaults (4 core checked)', d5.querySelectorAll('input[data-col]:checked').length === 4);
assert('compare: latency cols hidden by default', !d5.querySelector('input[data-col="p50"]').checked && !d5.querySelector('input[data-col="tps"]').checked);
assert('compare: show-errors button present', d5.querySelector('button#show-errors') !== null);
assert('compare: per-plugin columns built', d5.querySelector('input[data-col="plugin:smoke:score"]') !== null);

assert('compare multi-run: both models', d6.getElementById('compare').textContent.includes('m1') && d6.getElementById('compare').textContent.includes('m2'));
assert('compare multi-run: run column shown', d6.getElementById('compare').textContent.includes('r1') && d6.getElementById('compare').textContent.includes('r2'));
assert('compare multi-run: delete runs button present', d6.querySelector('button[data-delete-run]') !== null);

assert('run: no script error', !d7.title.startsWith('SCRIPT ERROR'));
assert('run: form visible', d7.getElementById('benchmark-form').style.display === 'block');
const opts = Array.from(d7.querySelectorAll('#models option')).map((o) => o.value);
assert('run: model options = 2', opts.length === 2 && opts.includes('qwen3.5:0.8b'));
assert('run: plugin checkboxes = 2', d7.querySelectorAll('#plugin-list input[type=checkbox]').length === 2);
assert('run: loading hidden', d7.getElementById('loading').style.display === 'none');

assert('resume: running -> progress visible', d8.getElementById('progress').style.display === 'block');
assert('resume: running -> poller scheduled', d8.defaultView.__intervals.length >= 1);
assert('resume: completed -> result shown', d9.getElementById('results').textContent.includes('Benchmark completed'));
assert('resume: unknown -> form visible', d10.getElementById('benchmark-form').style.display === 'block');
assert('resume: unknown -> key cleared', d10.defaultView.localStorage.getItem('local-ai-bench.active-run') === null);

assert('resume from ?run=: progress visible', d11.getElementById('progress').style.display === 'block');
assert('resume from ?run=: poller scheduled', d11.defaultView.__intervals.length >= 1);

console.log(process.exitCode ? '\nSome checks FAILED' : '\nAll page checks passed');
