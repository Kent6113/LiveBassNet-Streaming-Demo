// Password-protected content of the demo site (written by scripts/build_vault.mjs).
// The password never leaves the browser: it derives an AES-GCM key that decrypts
// the files' index and then each file on demand.  The derived key is kept for this
// browser tab (sessionStorage) so moving between pages does not ask again.

const KEYSTORE = 'lbn_vault_key_v1';
const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const toB64 = u8 => btoa(String.fromCharCode(...u8));

async function open(base, key) {
  const meta = await (await fetch(`${base}/vault.json`, { cache: 'no-cache' })).json();
  const dec = async u8 => new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: u8.slice(0, 12) }, key, u8.slice(12)));
  const files = JSON.parse(new TextDecoder().decode(await dec(b64(meta.index))));   // throws on a wrong key
  const urls = new Map();
  const bytes = async p => {
    const f = files[p]; if (!f) throw new Error(`not in the vault: ${p}`);
    return dec(new Uint8Array(await (await fetch(`${base}/b/${f.b}`)).arrayBuffer()));
  };
  return {
    has: p => p in files,
    list: prefix => Object.keys(files).filter(p => p.startsWith(prefix || '')),
    bytes,
    text: async p => new TextDecoder().decode(await bytes(p)),
    json: async p => JSON.parse(new TextDecoder().decode(await bytes(p))),
    url: async p => {
      if (!urls.has(p)) urls.set(p, bytes(p).then(u8 => URL.createObjectURL(new Blob([u8], { type: files[p].t }))));
      return urls.get(p);
    },
  };
}

async function derive(base, password) {
  const meta = await (await fetch(`${base}/vault.json`, { cache: 'no-cache' })).json();
  const pk = await crypto.subtle.importKey('raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveKey']);
  return crypto.subtle.deriveKey({ name: 'PBKDF2', salt: b64(meta.salt), iterations: meta.iter, hash: 'SHA-256' }, pk,
    { name: 'AES-GCM', length: 256 }, true, ['decrypt']);
}

// Resolve a vault for `base` (the folder holding vault.json), asking for the
// password in `mount` (an element) when this tab has no key yet.
export async function unlock(base, mount) {
  try {
    const raw = sessionStorage.getItem(KEYSTORE);
    if (raw) {
      const key = await crypto.subtle.importKey('raw', b64(raw), 'AES-GCM', true, ['decrypt']);
      return await open(base, key);
    }
  } catch { /* fall through to the form */ }
  return new Promise(resolve => {
    mount.innerHTML = `<form class="panel" id="vaultForm">
        <h2>This page is password protected</h2>
        <p class="muted">It holds unpublished results and audio from research datasets. Ask the author for the password.</p>
        <div class="row"><label for="vaultPw" class="small">Password</label>
          <input type="password" id="vaultPw" autocomplete="current-password" required>
          <button class="btn" type="submit">Open</button></div>
        <p class="small err" id="vaultErr" hidden></p></form>`;
    const form = mount.querySelector('#vaultForm'), err = mount.querySelector('#vaultErr');
    mount.querySelector('#vaultPw').focus();
    form.onsubmit = async e => {
      e.preventDefault(); err.hidden = true;
      const btn = form.querySelector('button'); btn.disabled = true; btn.textContent = 'Opening…';
      try {
        const key = await derive(base, mount.querySelector('#vaultPw').value);
        const v = await open(base, key);
        try { sessionStorage.setItem(KEYSTORE, toB64(new Uint8Array(await crypto.subtle.exportKey('raw', key)))); } catch {}
        mount.innerHTML = ''; resolve(v);
      } catch {
        err.textContent = 'That password does not open this page.'; err.hidden = false;
        btn.disabled = false; btn.textContent = 'Open';
      }
    };
  });
}

export function lock() { try { sessionStorage.removeItem(KEYSTORE); } catch {} }
