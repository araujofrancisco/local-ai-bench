// Shared UI helpers for OllamaBench pages: HTML escaping, toast notifications,
// and a promise-based confirmation modal. Imported by Astro page scripts and
// bundled by Vite at build time.

/** Escape a value for safe interpolation into HTML. */
export function esc(value) {
  return String(value ?? '').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
  );
}

const TOAST_COLORS = { info: '#3b82f6', success: '#10b981', error: '#ef4444' };

/** Show a transient toast notification. */
export function toast(message, type = 'info') {
  const el = document.createElement('div');
  el.textContent = message;
  Object.assign(el.style, {
    position: 'fixed',
    bottom: '1.25rem',
    right: '1.25rem',
    zIndex: '10000',
    background: TOAST_COLORS[type] || TOAST_COLORS.info,
    color: '#fff',
    padding: '0.75rem 1.25rem',
    borderRadius: '0.5rem',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    fontSize: '0.9rem',
    opacity: '0',
    transition: 'opacity 0.2s ease',
    pointerEvents: 'none',
  });
  document.body.appendChild(el);
  requestAnimationFrame(() => {
    el.style.opacity = '1';
  });
  setTimeout(() => {
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 250);
  }, 3500);
}

/**
 * Render a table header cell with an inline info hint.
 * The hint uses an 'i' glyph in a filled, distinct background so it stands out.
 * Pass `sortable` + `sortKey`/`sortDir` for click-to-sort support.
 */
export function thInfo(label, align = 'right', hint = '', { sortable = false, sortKey = null, sortDir = null } = {}) {
  const sort = sortable
    ? (sortKey && sortDir === 'desc' ? ' ▼' : sortKey && sortDir === 'asc' ? ' ▲' : '')
    : '';
  const hintHtml = hint
    ? `<span aria-label="${esc(hint)}" style="display:inline-flex;align-items:center;justify-content:center;width:1.15rem;height:1.15rem;border-radius:50%;color:#ffffff;font-size:.7rem;background:#475569;border:1px solid #64748b;margin-left:.35rem;cursor:help;">i</span>`
    : '';
  const cursor = sortable ? 'cursor:pointer' : 'cursor:default';
  return `<th title="${esc(hint || '')}" style="text-align:${align}; padding:1rem; color:#94a3b8; font-weight:500; ${cursor};" data-sort="${esc(sortKey || '')}">${esc(label)}${sort}${hintHtml}</th>`;
}

/** Format a number or null-ish value. `null` renders as '-'. */
export function formatNum(value, decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(decimals);
}

/** Color a pass-rate/percentage value (0-1 or 0-100 scale via opts). */
export function scoreColor(value) {
  const v = Number(value);
  if (Number.isNaN(v)) return 'inherit';
  return v >= 0.8 ? '#10b981' : v >= 0.5 ? '#f59e0b' : '#ef4444';
}

/**
 * Ask for confirmation with an accessible modal. Resolves to a boolean.
 * Closes on Cancel, overlay click, or Escape; focuses the confirm button.
 */
export function confirmModal({ title, body, confirmLabel = 'Delete', danger = true }) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.style.cssText =
      'position:fixed;inset:0;background:rgba(2,6,23,0.7);z-index:9999;' +
      'display:flex;align-items:center;justify-content:center;padding:1rem;';

    const dialog = document.createElement('div');
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    dialog.style.cssText =
      'background:#1e293b;border:1px solid #334155;border-radius:.75rem;' +
      'padding:1.5rem;max-width:420px;width:100%;';
    dialog.innerHTML = `
      <h3 style="color:#f1f5f9;font-size:1.125rem;margin-bottom:.75rem;">${esc(title)}</h3>
      ${body ? `<p style="color:#cbd5e1;font-size:.9rem;margin-bottom:1.25rem;">${esc(body)}</p>` : ''}
      <div style="display:flex;justify-content:flex-end;gap:.75rem;">
        <button data-cancel type="button" style="background:#334155;color:#e2e8f0;border:none;padding:.5rem 1rem;border-radius:.375rem;cursor:pointer;">Cancel</button>
        <button data-confirm type="button" style="background:${danger ? '#ef4444' : '#3b82f6'};color:#fff;border:none;padding:.5rem 1rem;border-radius:.375rem;cursor:pointer;font-weight:500;">${esc(confirmLabel)}</button>
      </div>`;

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    const onKey = (e) => {
      if (e.key === 'Escape') close(false);
    };
    const close = (result) => {
      document.removeEventListener('keydown', onKey);
      overlay.remove();
      resolve(result);
    };

    document.addEventListener('keydown', onKey);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) close(false);
    });
    dialog.querySelector('[data-cancel]').addEventListener('click', () => close(false));
    const confirmBtn = dialog.querySelector('[data-confirm]');
    confirmBtn.addEventListener('click', () => close(true));
    confirmBtn.focus();
  });
}
