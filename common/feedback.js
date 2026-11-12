// Sending ratings, listening-test results and (with consent) uploads to the
// feedback server set in ../config.js.  When no server is set or it cannot be
// reached, the result is queued in this browser (localStorage) and can be sent
// later or downloaded as a JSON file.

const QUEUE = 'lbn_feedback_queue_v1';
const api = () => ((window.LBN_CONFIG && window.LBN_CONFIG.api) || '').replace(/\/+$/, '');

function readQueue() { try { return JSON.parse(localStorage.getItem(QUEUE) || '[]'); } catch { return []; } }
function writeQueue(q) { try { localStorage.setItem(QUEUE, JSON.stringify(q)); return true; } catch { return false; } }

export function hasServer() { return !!api(); }
export function pending() { return readQueue(); }

async function post(kind, payload) {
  const r = await fetch(`${api()}/api/${kind}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok || !j.ok) throw new Error(j.error || `server answered ${r.status}`);
  return j;
}

// kind: "rating" | "test".  Resolves {where: "server" | "browser", id?}.
export async function send(kind, payload) {
  const item = { kind, payload: { ...payload, client_time: new Date().toISOString() } };
  if (api()) {
    try { const j = await post(kind, item.payload); return { where: 'server', id: j.id }; } catch { /* keep below */ }
  }
  const q = readQueue(); q.push(item);
  return { where: writeQueue(q) ? 'browser' : 'nowhere' };
}

// Resend everything queued; resolves the number still waiting.
export async function flush() {
  if (!api()) return readQueue().length;
  const left = [];
  for (const item of readQueue()) {
    try { await post(item.kind, item.payload); } catch { left.push(item); }
  }
  writeQueue(left);
  return left.length;
}

// Upload the listener's song, only after explicit consent.  meta must carry consent: true.
export async function upload(meta, file) {
  if (!api()) throw new Error('No feedback server is configured yet.');
  if (!meta || meta.consent !== true) throw new Error('Consent is required to upload audio.');
  const fd = new FormData();
  fd.append('meta', JSON.stringify(meta));
  if (file) fd.append('audio', file, file.name || 'audio');
  const r = await fetch(`${api()}/api/upload`, { method: 'POST', body: fd });
  const j = await r.json().catch(() => ({}));
  if (!r.ok || !j.ok) throw new Error(j.error || `server answered ${r.status}`);
  return j;
}

// Offer everything queued (or `data`) as a JSON file.
export function download(name, data) {
  const blob = new Blob([JSON.stringify(data ?? readQueue(), null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
